import logging
import os
import re
import subprocess

from stepin.script import spath


class Shell(object):
    def __init__(self, env_path='env', is_log=False):
        self.verbose = True
        self.is_log = is_log
        self.env_path = spath(env_path)
        self.env_bin_path = os.path.join(self.env_path, 'bin')
        exe_paths = {p.rstrip(os.path.sep) for p in re.split(r'\s*' + os.path.pathsep + r'\s*', os.environ['PATH'])}
        self.activated = self.env_bin_path in exe_paths

    def call(self, cmd, venv=False, checked=True, **args):
        if self.verbose:
            if self.is_log:
                logging.info('SHELL: ' + cmd)
            else:
                print(cmd)
        if venv:
            if not self.activated:
                if os.name == 'nt':
                    cmd = 'call ' + os.path.join(self.env_bin_path, 'activate.bat') + ' & ' + cmd
                else:
                    cmd = '. ' + os.path.join(self.env_bin_path, 'activate') + ' ; ' + cmd
        if checked:
            subprocess.check_call(cmd, shell=True, **args)
        else:
            subprocess.call(cmd, shell=True, **args)

    def ecall(self, cmd, checked=True, **args):
        self.call(cmd, checked=checked, venv=True, **args)

    def output(self, cmd, venv=False, **args):
        print(cmd)
        if venv:
            if not self.activated:
                if os.name == 'nt':
                    cmd = 'call ' + os.path.join(self.env_bin_path, 'activate.bat') + ' & ' + cmd
                else:
                    cmd = '. ' + os.path.join(self.env_bin_path, 'activate') + ' ; ' + cmd
        return subprocess.check_output(cmd, shell=True, **args)
