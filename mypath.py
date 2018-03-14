
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/media/eec/external/Databases/Segmentation/PASCAL/'
        elif database == 'sbd':
            return '/media/eec/external/Databases/Segmentation/PASCAL/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return 'models'
