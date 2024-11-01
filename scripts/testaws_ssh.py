# test ssh connection to aws instance

import os
import boto3
from paramiko import SSHClient,RSAKey
import pssh
from scp import SCPClient
import logging
from cProfile import Profile
from pstats import SortKey,Stats

from src.SSHSession import SSHSession

user = 'ec2-user'
host = 'ec2-35-182-19-168.ca-central-1.compute.amazonaws.com'
session = SSHSession(user,host)
command = 'sleep 4'
with Profile() as profile_command:
    ret = session.run_command(command)
    (
        Stats(profile_command)
        .strip_dirs()
        .sort_stats(SortKey.TIME)
        .print_stats(15)
    )
    print(ret)

with Profile() as profile_command:
    ret2 = session.run_command2(command)
    (
        Stats(profile_command)
        .strip_dirs()
        .sort_stats(SortKey.TIME)
        .print_stats(15)
    )
    print(ret)

pass
