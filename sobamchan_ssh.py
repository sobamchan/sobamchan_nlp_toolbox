from paramiko import SSHClient, AutoAddPolicy
import tempfile
import json
import os

HOST = os.environ.get('LAB_HOST')
USER = os.environ.get('LAB_USER')
PASS = os.environ.get('LAB_PASS')


class LabUploader(object):

    def __init__(self):
        pass

    @staticmethod
    def __up(save_path, fp):
        ssh = SSHClient()
        ssh.set_missing_host_key_policy(AutoAddPolicy())
        ssh.connect(HOST, 22, USER, PASS)
        sftp = ssh.open_sftp()
        fp.seek(0)
        sftp.put(fp.name, save_path)
        sftp.close()
        ssh.close()

    @staticmethod
    def upload(data, save_path, save_type):

        with tempfile.NamedTemporaryFile('w', dir='./') as fp:
            if save_type == 'json':
                json.dump(data, fp)
            if save_type == 'csv':
                data.to_csv(fp)
            if save_type == 'text':
                fp.write(data)

            LabUploader().__up(save_path, fp)

if __name__ == '__main__':
    LabUploader().upload('test\n',
            '/home/public/B4/takeshita/test.txt', 'text')
