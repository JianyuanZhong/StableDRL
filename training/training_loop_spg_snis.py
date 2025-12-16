# This file is modified by MAPLE research lab, based on the original code from https://github.com/NVlabs/edm

"""Main training loop for SPG (SNIS enhanced)."""

import os
import time
import psutil
import numpy as np
import torch
import sys
try:
    import dnnlib  # type: ignore
except Exception:  # pragma: no cover - fallback for static linters
    dnnlib = None  # type: ignore
try:
    import wandb
except ImportError:
    wandb = None # wandb is optional

from torch.optim.lr_scheduler import LambdaLR
try:
    from torch_utils import distributed as dist
    from torch_utils import misc
except ImportError:
    # ... (DummyDist and DummyMisc definitions remain the same)
    print("Warning: torch_utils not found. Distributed training may fail.")
    class DummyDist:
        def get_rank(self): return 0
        def get_world_size(self): return 1
        def print0(self, *args, **kwargs): print(*args, **kwargs)
        def get_accelerator(self): return None
    dist = DummyDist()
    class DummyMisc:
        def ddp_sync(self, module, sync):
            class DummyContextManager:
                def __enter__(self): return module
                def __exit__(self, exc_type, exc_val, exc_tb): pass
            return DummyContextManager()
        def count_parameters(self, module): return sum(p.numel() for p in module.parameters())
    misc = DummyMisc()

# Import the adapted SPG utilities
try:
    # Assuming llada_svpo.py is in the networks directory
    from networks.llada_svpo import (
        SPGConfig, generate_and_score_completions_spg, compute_loss_spg
    )
except ImportError:
    print("Error: llada_svpo utilities not found. Please ensure networks/llada_svpo.py is accessible.")
    # Define placeholders to prevent immediate crash, but training will fail.
    SPGConfig = lambda *args, **kwargs: None
    generate_and_score_completions_spg = lambda *args, **kwargs: None
    # Ensure placeholder returns a tensor that requires grad
    compute_loss_spg = lambda *args, **kwargs: {'loss': 0.0, 'loss_tensor': torch.tensor(0.0, requires_grad=True)}


#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',
    batch_size          = 512,
    data_loader_kwargs  = {},
    network_kwargs      = {},
    ref_network_kwargs  = {}, # Not used in SPG (beta=0)
    loss_kwargs        = {},
    optimizer_kwargs    = {},
    seed                = 0,
    total_steps         = 200000,
    loss_scaling        = 1,
    step_per_tick       = 50,
    snapshot_ticks      = 50,
    state_dump_ticks    = 500,
    cudnn_benchmark     = True,
    device              = torch.device('cuda'),
    grad_accumulation   = 1,
    lr_scheduler_kwargs = {},
    precision           = "fp16",
    resume_pt           = None,
    resume_state_dump   = None,
    resume_step         = 0,
    max_grad_norm       = 1000,
    val_ticks           = 5,
    skip_spike_grad     = 10e10,
    infer_kwargs        = {}, # Used here for generation configuration
    tokenizer_kwargs    = {},
    activation_checkpointing = 'whole_layer',
    training_state_dir  = None,
    # Added arguments for GRPO/SPG structure
    num_iterations      = 4, # Number of GRPO inner loop iterations
    random_masking      = True,
    *args, **kwargs
):
    dist.print0(f"Useless parameters: \n {args}\n {kwargs}")
    opts = {
        # ... (opts definition)
        "batch_size": batch_size,
        # ... (Omitted for brevity)
        "infer_kwargs": infer_kwargs,
        "num_iterations": num_iterations,
        "random_masking": random_masking,
    }

    # ... (Initialization: rank, seed, precision_dtype) ...
    rank = dist.get_rank()
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    if device.type == 'cuda':
        if precision == "fp16":
            precision_dtype = torch.float16
        elif precision == "bf16" and torch.cuda.is_bf16_supported():
            precision_dtype = torch.bfloat16
        else:
            precision_dtype = torch.float32
    else:
        precision_dtype = torch.float32


    # ... (Loading dataset, network) ...
    dist.print0('Loading dataset...')
    dataloader_iterator, reward_fn = dnnlib.util.construct_class_by_name(**data_loader_kwargs)
    dist.print0('Constructing network...')
    # network_kwargs now specifies LLaDASVPO (via the updated YAML)
    model = dnnlib.util.construct_class_by_name(**network_kwargs)
    model.eval().to(device)
    model_params = misc.count_parameters(model)
    if hasattr(model, 'model') and hasattr(model.model, 'set_activation_checkpointing'):
        model.model.set_activation_checkpointing(activation_checkpointing)

    # --- Robust Tokenizer Setup ---
    tokenizer = dnnlib.util.construct_class_by_name(**tokenizer_kwargs)
    model_name_path = tokenizer_kwargs.get('pretrained_model_name_or_path', '').lower()

    # Tokenizer setup for mask/pad tokens (Crucial for SPG)
    if 'gpt2' in model_name_path:
        dist.print0("Configuring tokenizer for GPT-2 style.")
        if getattr(tokenizer, 'mask_token_id', None) is None:
            dist.print0("Adding <MASK> token.")
            mask_token = "<MASK>"
            tokenizer.add_tokens([mask_token])
            tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
        if getattr(tokenizer, 'pad_token', None) is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Handle LLaDA/Llama style tokenizers
    elif 'llada' in model_name_path or 'llama' in model_name_path:
        dist.print0("Configuring tokenizer for LLaDA/Llama architecture.")
        # Ensure mask_token_id is set
        if getattr(tokenizer, 'mask_token_id', None) is None:
             # Default mask ID used in LLaDA pretraining/SPG paper
            default_mask_id = 126336
            dist.print0(f"Warning: mask_token_id not found. Setting to default: {default_mask_id}")
            tokenizer.mask_token_id = default_mask_id

        # Ensure pad_token_id is set
        if getattr(tokenizer, 'pad_token_id', None) is None:
            # In SPG framework, pad token is often set to mask token id for consistency
            tokenizer.pad_token_id = tokenizer.mask_token_id
            dist.print0(f"pad_token_id set to mask_token_id: {tokenizer.pad_token_id}")

    # Final check: SPG requires a mask token.
    if getattr(tokenizer, 'mask_token_id', None) is None:
         raise RuntimeError("Tokenizer configuration failed: mask_token_id is required for SPG and could not be determined.")
    # --- End Robust Tokenizer Setup ---


    # --- SPG Specific Initialization (Configuration) ---
    # Initialize SPGConfig based on provided kwargs (mapping user's config structure)
    spg_params = {}
    # Map parameters from loss_kwargs (e.g., spg_beta, spg_omega used in the original llada_svpo.py)
    spg_params['eubo_beta'] = loss_kwargs.get('spg_beta', 1.5)
    spg_params['mix_weight'] = loss_kwargs.get('spg_omega', 0.5)

    # Determine logp_estimation
    if 'logp_estimation' in loss_kwargs:
        spg_params['logp_estimation'] = loss_kwargs['logp_estimation']
    elif spg_params['mix_weight'] == 1.0:
        spg_params['logp_estimation'] = 'eubo'
    elif spg_params['mix_weight'] == 0.0:
        spg_params['logp_estimation'] = 'elbo'
    else:
        spg_params['logp_estimation'] = 'mix'

    # Map masking parameters
    spg_params['block_length'] = infer_kwargs.get('block_size', 32)
    # Use the mask_id identified from the tokenizer
    spg_params['mask_id'] = tokenizer.mask_token_id
    spg_params['num_t'] = loss_kwargs.get('num_mc_samples', 1) # Defaulting MC samples (num_t) to 1
    spg_params['p_mask_prompt'] = loss_kwargs.get('p_mask_perturb', 0.15)

    # New AIS/SNIS parameters
    spg_params['use_snis'] = loss_kwargs.get('use_snis', False)
    spg_params['ais_clip_iw'] = loss_kwargs.get('ais_clip_iw', 5.0)

    # Initialize SPGConfig
    spg_config = SPGConfig(**spg_params)

    # Initialize Generation Config (needed by generate_and_score_completions_spg)
    # Map parameters from infer_kwargs
    generation_config = {
        "max_completion_length": infer_kwargs.get('gen_length', 256),
        # Mapping max_steps (used in REB-PS) to diffusion_steps (used in LLaDA-style).
        "diffusion_steps": infer_kwargs.get('max_steps', 128),
        "temperature": infer_kwargs.get('temperature', 1.0),
        # Use block_size for consistency (key used in generate_and_score_completions_spg)
        "block_size": spg_config.block_length,
        "use_fp16": precision == "fp16" or precision == "bf16",
        "cfg_scale": infer_kwargs.get('cfg_scale', 0.0),
    }

    # ... (Setting up optimizer, scheduler, accelerator preparation, resume logic) ...
    dist.print0('Setting up optimizer...')
    # Disable foreach for Adam/AdamW to avoid mixed device/dtype grouping issues in multi-tensor ops
    try:
        _opt_class = optimizer_kwargs.get('class_name', '')
        if ('Adam' in _opt_class or 'AdamW' in _opt_class) and ('foreach' not in optimizer_kwargs):
            optimizer_kwargs['foreach'] = False
    except Exception:
        pass
    optimizer = dnnlib.util.construct_class_by_name(
        params=[p for p in model.parameters() if p.requires_grad],
        **optimizer_kwargs
    )
    scheduler = LambdaLR(optimizer, lr_lambda=dnnlib.util.construct_class_by_name(**lr_scheduler_kwargs))
    accelerator = dist.get_accelerator()
    if accelerator is not None:
        model, optimizer, dataloader_iterator, scheduler = accelerator.prepare(
           model, optimizer, dataloader_iterator, scheduler
        )

    # ... (Resume logic adaptation for GRPO structure) ...
    if resume_state_dump is not None and os.path.exists(resume_state_dump) and accelerator is not None:
        dist.print0(f"Resume from {resume_state_dump}")
        accelerator.load_state(resume_state_dump)
    if dataloader_iterator is not None:
        dataloader_iterator = iter(dataloader_iterator)
    if resume_state_dump is not None and os.path.exists(resume_state_dump) and resume_step > 0:
        print(f"Resume from step {resume_step}, skipping training data ...")
        # Calculate how many batches to skip. Generation happens every num_iterations steps.
        grpo_cycles_completed = resume_step // num_iterations
        batches_to_skip = grpo_cycles_completed * grad_accumulation
        
        if resume_step % num_iterations != 0:
             dist.print0("Warning: Resuming mid-GRPO cycle. Adjusting resume_step to the start of the cycle for stability.")
             # Align resume step to the start of the cycle
             resume_step = grpo_cycles_completed * num_iterations

        try:
            for _ in range(batches_to_skip):
                next(dataloader_iterator)
        except StopIteration:
            dist.print0("Warning: Dataset exhausted before reaching resume_step.")
        except TypeError:
            dist.print0("Warning: Dataloader is None during resume.")


    # ... (Training loop setup: cur_tick, training_step, wandb init) ...
    training_step = resume_step
    cur_tick = training_step // step_per_tick if step_per_tick > 0 else 0
    cur_nsamples = 0
    tick_start_step = training_step
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    text_table = None
    use_wandb = bool(wandb)

    # ... (WandB initialization - remains the same as provided in the prompt) ...
    if rank == 0:
        if use_wandb:
             # ... (WandB setup remains the same as provided in the prompt)
            run_name = ':'.join(run_dir.split('/')[-2:])
            wandb_kwargs = dict(
                entity='jianyuan',
                project="rl-discrete-diffusion-TRVPO",
                name=run_name,
                dir=run_dir,
                config=opts,
                mode='online',
            )
            os.environ.setdefault('WANDB_CONSOLE', 'off')
            os.environ.setdefault('WANDB_SILENT', 'true')
            class _StderrTtyShim:
                # ... (Shim implementation)
                def __init__(self, underlying):
                    self._underlying = underlying
                def isatty(self):
                    return False
                def write(self, data):
                    try:
                        return self._underlying.write(data)
                    except Exception:
                        try:
                            return getattr(self._underlying, 'error', lambda *_args, **_kwargs: None)(str(data).rstrip('\n'))
                        except Exception:
                            return None
                def flush(self):
                    try:
                        return self._underlying.flush()
                    except Exception:
                        return None
                def __getattr__(self, name):
                    return getattr(self._underlying, name)
            _stderr_backup = sys.stderr
            try:
                sys.stderr = _StderrTtyShim(_stderr_backup)
                wandb.init(**wandb_kwargs)
            except Exception as e:
                 print(f"WandB initialization failed: {e}")
                 use_wandb = False
            finally:
                sys.stderr = _stderr_backup

        eval_dir = os.path.join(run_dir, 'evaluations')
        os.makedirs(eval_dir, exist_ok=True)
        if use_wandb:
            text_table = wandb.Table(columns=['step', 'prompt', 'response'])


    dist.print0(f'Training for {total_steps} steps in {precision_dtype}...')
    dist.print0(f"Model with Param: {model_params}")
    dist.print0(f"Using GRPO with {num_iterations} iterations.")
    dist.print0()

    num_generations = infer_kwargs.get('num_generations', 1)
    repeat_times = infer_kwargs.get('repeat_times', 1)
    total_generations_per_prompt = num_generations * repeat_times

    # Total samples processed per GRPO cycle generation phase
    batch_total = batch_size * dist.get_world_size() * grad_accumulation * total_generations_per_prompt

    # Effective gain for gradient accumulation
    effective_gain = loss_scaling / grad_accumulation
    dist.print0(f"Gradient Accumulation: {grad_accumulation}, Effective Gain (Loss Scale): {effective_gain}")

    # Timers for benchmarking
    timer_stats = {'data': 0.0, 'gen': 0.0, 'loss': 0.0, 'backward': 0.0, 'optim': 0.0}
    timer_counts = {'data': 0, 'gen': 0, 'loss': 0, 'backward': 0, 'optim': 0}

    done = False
    # Buffer for GRPO iterations, keyed by gradient accumulation index
    buffered_inputs = {}
    current_batch = None # Store the current batch metadata for logging

    while not done:
        if rank == 0 and not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)

        optimizer.zero_grad(set_to_none=True)
        all_loss_log_kwargs = []

        # Determine the current iteration index within the GRPO cycle
        current_itr_idx = training_step % num_iterations

        for round_idx in range(grad_accumulation):
            buffer_key = round_idx
            is_last_acc_step = (round_idx == grad_accumulation - 1)

            # Use DDP sync appropriately
            with misc.ddp_sync(model, sync=is_last_acc_step):

                if current_itr_idx == 0:
                    # Start of a new GRPO cycle: Generate samples and score
                    model.eval()
                    if rank == 0:
                        print(f"Step {training_step}: Start Sampling and Scoring (GRPO cycle start)...")

                    try:
                        t_start_data = time.time()
                        current_batch = next(dataloader_iterator)
                        timer_stats['data'] += time.time() - t_start_data
                        timer_counts['data'] += 1
                    except (StopIteration, TypeError):
                        dist.print0("Dataset exhausted or Dataloader error, terminating training.")
                        done = True
                        break

                    # Call the adapted generation and scoring function
                    # Precision is managed by Accelerate.
                    t_start_gen = time.time()
                    inputs = generate_and_score_completions_spg(
                        model=model,
                        tokenizer=tokenizer,
                        inputs_batch=current_batch,
                        reward_func=reward_fn,
                        device=device,
                        accelerator=accelerator,
                        args=spg_config,
                        num_iterations=num_iterations,
                        generation_config=generation_config,
                        num_generations=total_generations_per_prompt,
                        random_masking=random_masking
                    )
                    timer_stats['gen'] += time.time() - t_start_gen
                    timer_counts['gen'] += 1

                    if inputs is None:
                        dist.print0(f"Warning: Empty or failed batch encountered during generation (Step {training_step}, Round {round_idx}). Skipping.")
                        buffered_inputs[buffer_key] = None # Mark this round as failed
                        continue

                    buffered_inputs[buffer_key] = inputs
                    torch.cuda.empty_cache()

                else:
                    # Subsequent GRPO iterations: Reuse buffered inputs
                    if buffer_key not in buffered_inputs or buffered_inputs[buffer_key] is None:
                        # Skip if the generation failed in the first iteration for this round
                        continue
                    inputs = buffered_inputs[buffer_key]

                if done: break

                if accelerator:
                    accelerator.wait_for_everyone()

                if rank == 0 and current_itr_idx > 0:
                    print(f"Step {training_step}: Start Loss Calc (GRPO iteration {current_itr_idx+1}/{num_iterations})...")

                # Compute Loss and perform backward pass
                model.train()

                # FIX: Removed manual torch.autocast; rely on Accelerate's precision management
                t_start_loss = time.time()
                loss_log_kwargs = compute_loss_spg(
                    model=model,
                    inputs=inputs,
                    current_itr_idx=current_itr_idx,
                    args=spg_config,
                    accelerator=accelerator,
                    random_masking=random_masking # MODIFICATION: Pass the random_masking flag
                )
                timer_stats['loss'] += time.time() - t_start_loss
                timer_counts['loss'] += 1

                # Perform backward pass
                if 'loss_tensor' in loss_log_kwargs:
                    loss_tensor = loss_log_kwargs.pop('loss_tensor')
                    
                    # Use the effective_gain to scale loss for accumulation
                    # Note: Accelerate handles scaling for mixed precision, but we must manually divide by accumulation steps
                    scaled_loss = loss_tensor / grad_accumulation 
                    
                    t_start_bwd = time.time()
                    if accelerator is not None:
                        accelerator.backward(scaled_loss)
                    else:
                        scaled_loss.backward()
                    timer_stats['backward'] += time.time() - t_start_bwd
                    timer_counts['backward'] += 1

                    all_loss_log_kwargs.append(loss_log_kwargs)
                else:
                    dist.print0(f"Warning: Loss tensor missing at Step {training_step}, Round {round_idx}.")


                torch.cuda.empty_cache()

            if done: break

            # Validation logging (Adapted for the new structure)
            # We only log validation at the start of the GRPO cycle (current_itr_idx == 0).
            if cur_tick % val_ticks == 0 and rank == 0 and round_idx == 0 and current_itr_idx == 0:
                if inputs and 'completion_ids' in inputs and current_batch and 'problems' in current_batch and text_table is not None:
                    # ... (Adaptation of validation logging) ...
                    try:
                        if 'answers' in current_batch and len(current_batch['answers']) > 0:
                            text_inputs = current_batch['problems'][0] + '\n\n' + current_batch['answers'][0]
                        elif len(current_batch['problems']) > 0:
                            text_inputs = current_batch['problems'][0] + '\n\n'
                        else:
                            text_inputs = "N/A"

                        # Decode the completions generated in this round
                        dec_inputs = inputs['completion_ids'].detach().cpu().tolist()
                        text_responses = "\n***\n".join(
                            tokenizer.batch_decode(dec_inputs, skip_special_tokens=True)
                        )
                        if use_wandb:
                            text_table.add_data(str(training_step), text_inputs, text_responses)

                        # Log to file
                        if eval_dir:
                             with open(os.path.join(eval_dir, f'evaluate_{training_step}.txt'), 'w', encoding='utf-8') as f:
                                f.write(text_inputs + '\n' + '=' * 20 + '\n' + text_responses)
                    except Exception as e:
                        dist.print0(f"Warning: Validation logging failed: {e}")


            if accelerator:
                accelerator.wait_for_everyone()

            # Cleanup buffer references at the end of the GRPO cycle to free memory
            if (training_step + 1) % num_iterations == 0:
                if buffer_key in buffered_inputs:
                    # Clean up tensors in the buffer before deleting the entry
                    if buffered_inputs[buffer_key]:
                         del buffered_inputs[buffer_key]
                torch.cuda.empty_cache()

        if done: break

        # --- Optimization Step ---

        if not all_loss_log_kwargs:
            print(f"Warning: No loss calculated at step {training_step}. Skipping optimization step.")
            # If generation failed (itr=0), we must advance the step counter to avoid deadlock.
            if current_itr_idx == 0:
                 # Skip the remaining iterations of the cycle
                 training_step += num_iterations
            else:
                 training_step += 1
            continue

        # Aggregate loss logs (Average over accumulation steps)
        loss_log_kwargs = {}
        if all_loss_log_kwargs and all_loss_log_kwargs[0]:
            for k in all_loss_log_kwargs[0].keys():
                # Average the logged metrics
                loss_log_kwargs[k] = sum([item[k] for item in all_loss_log_kwargs if k in item]) / len(all_loss_log_kwargs)

        # ... (Gradient processing: nan_to_num, clipping, grad_norm calculation) ...
        # Removed manual nan_to_num on param.grad as it can interfere with FSDP/Accelerate
        # for param in model.parameters():
        #     if param.grad is not None:
        #         torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)

        if accelerator:
            _grad_norm = accelerator.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )
        else:
            _grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)


        try:
            grad_norm_tensor = torch.tensor([float(_grad_norm)], device=device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Use accelerator.reduce for consistency if available
                if accelerator:
                    grad_norm_tensor = accelerator.reduce(grad_norm_tensor, reduction="max")
                else:
                    torch.distributed.all_reduce(grad_norm_tensor, op=torch.distributed.ReduceOp.MAX)
            grad_norm_synced = float(grad_norm_tensor.item())
        except Exception:
            grad_norm_synced = float(_grad_norm)

        # Optimization step
        t_start_optim = time.time()
        optimizer.step()
        timer_stats['optim'] += time.time() - t_start_optim
        timer_counts['optim'] += 1
        # scheduler.step(training_step) # Update scheduler based on training_step - MOVED below


        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        if rank == 0 and use_wandb: # and current_itr_idx == 0: # Log every step
            log_data = {
                'lr': current_lr,
                'grad_norm': grad_norm_synced,
            }
            for k, v in loss_log_kwargs.items():
                try:
                    log_data[k] = float(v)
                except (ValueError, TypeError):
                    continue
            wandb.log(log_data, step=training_step)

        # Update counters
        # cur_nsamples is updated based on actual generation steps (itr=0)
        if current_itr_idx == 0:
            cur_nsamples += batch_total

        done = (training_step >= total_steps)
        
        # Scheduler step should happen after the step update if it takes the current step as input
        # But more importantly, training_step should increment per gradient update
        scheduler.step(training_step) 
        
        training_step += 1

        # Tick management
        if (not done) and (training_step < tick_start_step + step_per_tick):
            continue

        # ... (Tick logging, checkpointing, maintenance time update) ...
        tick_end_time = time.time()
        # ... (Fields definition for logging) ...
        fields = {
            'tick': cur_tick,
            'step': training_step,
            # ... (Other fields omitted for brevity) ...
            'grad_norm': grad_norm_synced,
            'lr': current_lr,
            't_data': timer_stats['data'] / max(1, timer_counts['data']),
            't_gen': timer_stats['gen'] / max(1, timer_counts['gen']),
            't_loss': timer_stats['loss'] / max(1, timer_counts['loss']),
            't_bwd': timer_stats['backward'] / max(1, timer_counts['backward']),
            't_opt': timer_stats['optim'] / max(1, timer_counts['optim']),
            **loss_log_kwargs,
        }
        
        # Reset timers
        for k in timer_stats:
            timer_stats[k] = 0.0
            timer_counts[k] = 0

        # Log every tick
        if is_last_acc_step: # and current_itr_idx == 0:
            for key, value in fields.items():
                dist.print0(f"{key} {value}", end='\t')
            dist.print0()

        if device.type == 'cuda':
             torch.cuda.reset_peak_memory_stats()

        # ... (Console logging) ...

        # Checkpointing - Save only at the end of the GRPO cycle
        is_end_of_cycle = (training_step % num_iterations == 0) or num_iterations == 1
        if accelerator and (cur_tick % snapshot_ticks == 0) and is_end_of_cycle:
            try:
                state_dict = accelerator.get_state_dict(model)
                if training_state_dir:
                    save_path = os.path.join(training_state_dir, f'training-state-{training_step:06d}')
                    accelerator.save_state(save_path)
                if rank == 0:
                    save_path = os.path.join(run_dir, f'ckpt-{training_step:06d}')
                    accelerator.unwrap_model(model).save_pretrained(
                        save_path, state_dict=state_dict, safe_serialization=True
                    )
            except Exception as e:
                dist.print0(f"Warning: Checkpoint saving failed: {e}")

        if accelerator:
            accelerator.wait_for_everyone()

        # Update tick counters
        tick_end_time = time.time()
        maintenance_time = tick_end_time - tick_start_time
        cur_tick += 1
        cur_nsamples = 0
        tick_start_step = training_step
        tick_start_time = time.time()
        if done:
            break

    # ... (Final logging and exit) ...
    if rank == 0 and use_wandb and text_table:
        wandb.log({
            'text_response': text_table,
        })
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------

