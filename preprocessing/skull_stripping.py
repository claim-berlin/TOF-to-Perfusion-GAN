#!/usr/bin/env python

from __future__ import print_function

# -------------------------------- SynthStrip --------------------------------

# This is a short wrapper script around a Docker-ized version of SynthStrip.
# The aim of this script is to minimize the effort required to use the
# SynthStrip docker container by automatically mounting any necessary input
# and output files paths. This script can be used with the same syntax as the
# default FreeSurfer `mri_synthstrip` command (use the --help flag for more info).
# Upon first use, the relevant docker image will be automatically pulled from
# DockerHub. To use a different SynthStrip version, update the variable below.
version = '1.1'

# ----------------------------------------------------------------------------

import os
import sys
import subprocess
import shutil


TOP_PATH = "/Data/Heidelberg/"        
SUBJ_LIST = ['0027', '0029', '0034', '0046', '0049', '0059', '0062', '0071', 
             '0074', '0075', '0083', '0088', '0092', '0096', '0105', '0124', 
             '0129', '0134', '0136', '0162', '0170', '0178', '0179', '0180', 
             '0184', '0188', '0194', '0195', '0196', '0197', '0199', '0202', 
             '0208', '0209', '0210', '0211', '0214', '0215', '0216', '0218', 
             '0219', '0220', '0228', '0233', '0237', '0240', '0241', '0242', 
             '0245', '0255', '0258', '0263', '0266', '0267', '0270', '0272', 
             '0273', '0275', '0277', '0287', '0288', '0290', '0293', '0299', 
             '0302', '0305', '0306', '0307', '0308', '0310', '0313', '0314', 
             '0317', '0318', '0320', '0321', '0325', '0329', '0331', '0332', 
             '0333', '0334', '0335', '0337', '0342', '0344', '0345', '0346', 
             '0348', '0350', '0352', '0355', '0356', '0357', '0360', '0362', 
             '0363', '0365', '0367', '0368', '0371', '0373', '0378', '0380', 
             '0382', '0383', '0388', '0390', '0391', '0393', '0394', '0395', 
             '0396', '0397', '0400', '0402', '0403', '0407', '0410', '0417', 
             '0420', '0421', '0430', '0432', '0433', '0435', '0442', '0446', 
             '0451', '0460', '0464', '0465', '0466', '0473', '0483', '0488', 
             '0492', '0494', '0495', '0496', '0513', '0515', '0519', '0528', 
             '0534', '0547', '0565', '0574', '0591', '0604', '0638', '0639', 
             '0661', '0668', '0676', '0681', '0686', '0690', '0695', '0704', 
             '0705', '0718', '0720', '0729', '0731', '0764', '0766', '0767', 
             '0774', '0785', '0792', '0826', '0835', '0839', '0843', '0887', 
             '0907', '0908', '0910', '0933', '0960', '0979', '1015', '1024', 
             '1029', '1038', '1081', '1112', '1139', '1143', '1153', '1169', 
             '1182', '1213', '1362', '1414', '1491', '1508', '1514', '1552']
SEQS = ["TOF"]

# Sanity check on env
if shutil.which('docker') is None:
    print('Cannot find docker in PATH. Make sure it is installed.')
    exit(1)

# Since we're wrapping a Docker image, we want to get the full paths of all input and output
# files so that we can mount their corresponding paths. Tedious, but a fine option for now...

for pat_id in SUBJ_LIST:
    for seq in SEQS:
        print(pat_id)
        input_file = "{}result_folder/final/{}_{}.nii.gz".format(TOP_PATH, pat_id, seq)
        output_file = "{}result_folder/TOF_masks/{}_{}.nii.gz".format(TOP_PATH, pat_id, seq)
        mask_file = "{}result_folder/TOF_masks/{}_{}_mask.nii.gz".format(TOP_PATH, pat_id, seq)
        
        flags = ['-i', '--input', '-o', '--output', '-m', '--mask', '--model']
        input_arg = ['', '-i', input_file, '-o', output_file, '-m', mask_file]
        
        # Loop through the arguments and expand any necessary paths
        idx = 1
        args = []
        paths = []
        while idx < len(input_arg):
            arg = input_arg[idx]
            args.append(arg)
            if arg in flags:
                idx += 1
                path = os.path.realpath(os.path.abspath(input_arg[idx]))
                args.append(path)
                paths.append(path)
            idx += 1
        args = ' '.join(args)
        print(args)

        # Get the unique mount points
        mounts = list(set([os.path.dirname(p) for p in paths]))
        mounts = ' '.join(['-v %s:%s' % (p, p) for p in mounts])

        print('Running SynthStrip from Docker')

        # Get image tag
        image = 'freesurfer/synthstrip:' + version

        # Let's check to see if we have this container on the system
        proc = subprocess.Popen('docker images -q %s' % image,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                universal_newlines=True)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            print(stderr)
            print('Error running docker command. Make sure Docker is installed.')
            exit(proc.returncode)

        # If not, let's download it. Normally, docker run will do this automatically,
        # but we're trying to be transparent here...
        if not stdout:
            print('Docker image %s is not installed. Downloading now. This only needs to be done once.' % image)
            proc = subprocess.Popen('docker pull %s' % image, shell=True)
            proc.communicate()
            if proc.returncode != 0:
                print('Error running docker pull.')
                exit(proc.returncode)

        # Go ahead and run the entry point
        command = 'docker run %s %s %s' % (mounts, image, args)
        proc = subprocess.Popen(command, shell=True)
        proc.communicate()
        if proc.returncode == 137:
            print('Container ran out of memory, try increasing RAM in Docker preferences.')
            exit(proc.returncode)
        if proc.returncode != 0:
            print('Error running image.')
            exit(proc.returncode)
