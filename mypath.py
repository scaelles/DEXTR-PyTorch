
class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/csergi/scratch2/Databases/PASCAL2012/'
        return None

    @staticmethod
    def models_dir():
        return 'models'
