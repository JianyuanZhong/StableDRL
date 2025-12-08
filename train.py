import os
import re
import json
import yaml
import click
import dnnlib
from datetime import datetime
from torch_utils import distributed as dist

def CommandWithConfigFile(config_file_param_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_file_param_name]
            if config_file is not None:
                with open(config_file, encoding='utf-8') as f:
                    config_data = yaml.load(f, Loader=yaml.FullLoader)
                    for key, value in config_data.items():
                        ctx.params[key] = value
            return super(CustomCommandClass, self).invoke(ctx)

    return CustomCommandClass

#----------------------------------------------------------------------------

@click.command(cls=CommandWithConfigFile("config"))
@click.option("--config",    help="config file path",    type=click.Path(exists=True))
@click.option('--fsdp', is_flag=True, help='use fsdp')
@click.option('--resume-from', 'resume_from', default=None, type=click.Path(exists=True), help='Resume training from an existing run folder (path to run dir)')
def main(**kwargs):
    kwargs.pop("config")
    _use_fsdp = kwargs.pop("fsdp", False)
    resume_from = kwargs.pop("resume_from", None)
    training_args = dnnlib.EasyDict(kwargs.pop("training_args"))
    opts = dnnlib.EasyDict(kwargs)
    # torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0("Distributed initialized.")

    try:
        training_args.batch_size = opts.data_loader_kwargs.get('batch_size', 1)
    except Exception:
        dist.print0("Batch size not set?")

    # Handle resume-from mode vs starting a new run
    resuming_mode = resume_from is not None
    if resuming_mode:
        # Use the provided existing run directory, do not mutate it
        training_args.run_dir = resume_from
        # Load prior training options if present to recover training_state_dir
        prior_opts_path = os.path.join(training_args.run_dir, 'training_options.json')
        prior_training_state_dir = None
        if os.path.exists(prior_opts_path):
            try:
                with open(prior_opts_path, 'rt', encoding='utf-8') as f:
                    prior_dump = json.load(f)
                prior_training_state_dir = prior_dump.get('training_state_dir', None)
            except Exception:
                prior_training_state_dir = None
        # Fallback: default to the parent date folder STATES-*
        if prior_training_state_dir is None:
            parent_dir = os.path.dirname(training_args.run_dir)
            # Prefer an existing STATES-* folder
            candidates = [d for d in os.listdir(parent_dir) if d.startswith('STATES-') and os.path.isdir(os.path.join(parent_dir, d))]
            training_args.training_state_dir = os.path.join(parent_dir, candidates[0]) if len(candidates) > 0 else os.path.join(parent_dir, f'STATES-{os.environ.get("SLURM_JOB_ID") or "tmp"}')
        else:
            training_args.training_state_dir = prior_training_state_dir
        dist.print0(f"[Resume] Using training_state_dir: {training_args.training_state_dir}")

        # Determine last step and state/ckpt paths
        latest_step = 0
        resume_state_dump = None
        if os.path.isdir(training_args.training_state_dir):
            state_dirs = [x for x in os.listdir(training_args.training_state_dir) if x.startswith('training-state-')]
            for sd in state_dirs:
                m = re.match(r'^training-state-(\d+)$', sd)
                if m:
                    latest_step = max(latest_step, int(m.group(1)))
            if latest_step > 0:
                resume_state_dump = os.path.join(training_args.training_state_dir, f'training-state-{latest_step:06d}')

        # If no state dump found, still try to resume model weights from latest ckpt
        ckpt_step = 0
        ckpt_dirs = [x for x in os.listdir(training_args.run_dir) if x.startswith('ckpt-') and os.path.isdir(os.path.join(training_args.run_dir, x))]
        for cd in ckpt_dirs:
            m = re.match(r'^ckpt-(\d+)$', cd)
            if m:
                ckpt_step = max(ckpt_step, int(m.group(1)))
        # Prefer state-dump step for resume_step; otherwise fall back to ckpt step
        resume_step = latest_step if latest_step > 0 else ckpt_step

        # If we have a ckpt dir, load the model from it by overriding the network pretrained path
        if ckpt_step > 0:
            last_ckpt_dir = os.path.join(training_args.run_dir, f'ckpt-{ckpt_step:06d}')
            if isinstance(opts.network_kwargs, dict):
                opts.network_kwargs['pretrained_model_name_or_path'] = last_ckpt_dir
            else:
                # Fallback in case of EasyDict
                opts.network_kwargs.pretrained_model_name_or_path = last_ckpt_dir
            dist.print0(f"[Resume] Loading model from checkpoint: {last_ckpt_dir}")

        # Inject resume args
        training_args.resume_state_dump = resume_state_dump
        training_args.resume_step = int(resume_step)
        # Keep batch size, precision, etc. from the new config, but reuse run_dir
    else:
        # Description string.
        dtype_str = training_args.precision
        desc = f'gpus{dist.get_world_size():d}-batch{training_args.batch_size:d}-{dtype_str:s}'
        if opts.desc is not None:
            desc += f'-{opts.pop("desc")}'
            
        date = datetime.now().strftime("%Y-%m-%d")
        training_args.run_dir = os.path.join(training_args.run_dir, date)
        training_args.training_state_dir = os.path.join(training_args.run_dir, f'STATES-{os.environ.get("SLURM_JOB_ID") or "tmp"}')
        dist.print0(f"Save training_args at {training_args.training_state_dir}")
        # Pick output directory.
        if dist.get_rank() != 0:
            training_args.run_dir = None
            
        else:
            prev_run_dirs = []
            if os.path.isdir(training_args.run_dir):
                prev_run_dirs = [x for x in os.listdir(training_args.run_dir) if os.path.isdir(os.path.join(training_args.run_dir, x))]
            prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
            prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
            slurm_job_id = int(os.environ.get("SLURM_JOB_ID")) - 1 if os.environ.get("SLURM_JOB_ID", None) else -1
            cur_run_id =  max(prev_run_ids + [slurm_job_id], default=-1) + 1
            training_args.run_dir = os.path.join(training_args.run_dir, f'{cur_run_id:05d}-{desc}')
            assert not os.path.exists(training_args.run_dir)

    # Print options.
    dump_dict = opts.copy()
    dump_dict.update(training_args)
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(dump_dict, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {training_args.run_dir}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {training_args.batch_size}')
    dist.print0(f'Precision:               {training_args.precision}')
    dist.print0()

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(training_args.run_dir, exist_ok=True)
        # Only write options file when starting a new run, avoid clobbering prior config when resuming
        if not resuming_mode:
            with open(os.path.join(training_args.run_dir, 'training_options.json'), 'wt', encoding='utf-8') as f:
                json.dump(dump_dict, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(training_args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    dnnlib.util.call_func_by_name(
        **opts,
        **training_args,
    )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

