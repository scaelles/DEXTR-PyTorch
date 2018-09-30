#!/usr/bin/python
import sys
import os
import argparse
import itertools
import json
import random
import subprocess


def parse_json(grid):
    perms = list(itertools.product(*grid.values()))
    random.shuffle(perms)
    commands = []
    n_gpus = []

    for p in perms:

        argstr = ""
        for i, k in enumerate(grid.keys()):
            if type(p[i]) is int or type(p[i]) is float:
                v = str(p[i])
            else:
                assert '"' not in p[i]
                v = '"%s"' % p[i]
            key = str(k)
            if len(key) > 2:
                argstr += " --%s %s" % (str(k), v)
            else:
                argstr += " -%s %s" % (str(k), v)
        commands.append(argstr)

    return commands, n_gpus


COMMANDS = ['run', 'info', 'kill']

if (len(sys.argv) < 2) or (sys.argv[1] not in COMMANDS):
    print('Error: recognized commands: %s' % ' '.join(COMMANDS))
    exit(1)
COMMAND = sys.argv[1]

parser = argparse.ArgumentParser(description='slurm jobs')

parser.add_argument('--name', type=str, default=None)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nodes', type=int, default=1)
parser.add_argument('--sweep', type=str, default='')
parser.add_argument('--exclude', type=str, default=None)
parser.add_argument('--partition', type=str, default="uninterrupted")
parser.add_argument('--time', type=int, default=4320,
                    help='Time to allocate in the cluster: default 3 days')

if sys.argv[-1][-3:] == '.sh':
    to_parse = sys.argv[2:-1]
    EXPERIMENT = ''
    SCRIPT = sys.argv[-1]
else:
    to_parse = sys.argv[2:]
    EXPERIMENT = sys.argv[-1]
    SCRIPT = 'submit_job.sh'

args = parser.parse_args(to_parse)

if args.sweep[-5:] != '.json':
    args.sweep = os.path.join(args.sweep, args.sweep + '.json')

if args.name is None and args.sweep != '':
    args.name = args.sweep.split('.')[0].split('/')[-1]
    print('Name: ' + args.name)


if COMMAND == 'info':
    STRING = 'squeue -u `whoami` --name %s' % args.name
    subprocess.call(STRING, shell=True)

if COMMAND == 'kill':
    STRING = 'scancel --name %s' % args.name
    subprocess.call(STRING, shell=True)

if COMMAND == 'run':
    STRING = 'sbatch --partition={} --ntasks-per-node=1 --cpus-per-task=10 --open-mode=append '.format(args.partition)
    STRING += '--time=%s ' % args.time
    STRING += '--nodes=%s ' % args.nodes
    STRING += '--job-name=%s ' % args.name
    STRING += '--output=/checkpoint/%%u/logs/%s_%%j.out ' % args.name
    STRING += '--error=/checkpoint/%%u/logs/%s_%%j.out ' % args.name

    if not os.path.isfile(args.sweep):
        print('Error: sweep file does not exist')

    with open(args.sweep, 'r') as f:
        [sweep_commands, n_gpus] = parse_json(json.loads(f.read()))

    if args.ngpu > 0:
        STRING += '--gres=gpu:%s ' % args.ngpu

    if args.exclude is not None:
        STRING += '--exclude=%s ' % args.exclude

    STRING += '--wrap="srun --label %s' % SCRIPT.replace('"', '\\"')
    STRING += ' ' + EXPERIMENT + ' '

    print("Submitting {} jobs".format(len(sweep_commands)))
    n_req_gpus = args.ngpu * len(sweep_commands) if not n_gpus else sum(n_gpus)
    print("Requesting in total {} GPUs".format(n_req_gpus))

    for i, command in enumerate(sweep_commands):
        if n_gpus:
            SUBM_STRING = STRING.replace('--gres=gpu:8', '--gres=gpu:{}'.format(str(n_gpus[i])))
        else:
            SUBM_STRING = STRING
        print(SUBM_STRING + command.replace('"', '\\"') + '"')
        subprocess.call(SUBM_STRING + command.replace('"', '\\"') + '"', shell=True)