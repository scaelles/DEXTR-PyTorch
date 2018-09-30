
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/path/to/PASCAL/'  # folder that contains VOCdevkit/.

        elif database == 'sbd':
            return '/path/to/SBD/'  # folder that contains benchmark_RELEASE
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models/'

    @staticmethod
    def save_dir_root():
        return '/path/to/store/experiment/results/'
