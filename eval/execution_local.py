import contextlib
import faulthandler
import io
import multiprocessing
import os
import platform
import signal
import tempfile
from typing import Dict, Optional


def check_correctness(
    task_id: str,
    sample: dict,
    language_type: str,
    timeout: float = 3.0,
    tmp_dir: str = None,
    completion_id: Optional[int] = None,
) -> Dict:
    """
    Evaluate functional correctness for Python completions by executing provided test_code.
    Other languages are not supported in this local evaluator.
    """

    def unsafe_execute(_: Optional[str] = None):
        if "python" not in language_type.lower():
            result.append("failed: unsupported language")
            return
        with create_tempdir():
            # keep handles to restore
            import os as _os
            import shutil as _shutil
            rmtree = _shutil.rmtree
            rmdir = _os.rmdir
            chdir = _os.chdir
            orig_unlink, orig_rmdir, orig_rmtree = _os.unlink, _os.rmdir, _shutil.rmtree
            try:
                reliability_guard()
                exec_globals: Dict[str, object] = {}
                code = sample.get("test_code", "")
                with swallow_io():
                    with time_limit(timeout):
                        exec(code, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except AssertionError:
                result.append("failed: AssertionError")
            except BaseException as e:
                print("-" * 10 + 'code' + "-" * 10)
                print(code)
                print("-" * 10 + 'code' + "-" * 10)
                import traceback
                traceback.print_exc()
                result.append(f"failed: {e}")
            finally:
                _os.unlink, _os.rmdir, _shutil.rmtree = orig_unlink, orig_rmdir, orig_rmtree
                _shutil.rmtree = rmtree
                _os.rmdir = rmdir
                _os.chdir = chdir

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=unsafe_execute, args=(tmp_dir,))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result.append("timed out")
    return {
        "task_id": task_id,
        "completion_id": completion_id,
        "result": result[0],
        "passed": result[0] == "passed",
        "finish": -1 if "finish" not in sample else sample["finish"],
        "code": sample.get("test_code", ""),
    }


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    os.environ['OMP_NUM_THREADS'] = '1'

    def _disabled(*a, **kw):
        raise RuntimeError("Restricted system call")

    os.system = _disabled
    os.kill = _disabled
    os.setuid = _disabled
    os.fork = _disabled

    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None