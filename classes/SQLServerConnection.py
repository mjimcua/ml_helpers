import sqlalchemy
import urllib


class SQLServerConnection:

    def __init__(self, driver, server, port, database, charset='utf8'):

        params = 'DRIVER={' + driver + '};'\
            'SERVER=' + server + ';' \
            'PORT=' + str(port) + ';' \
            'DATABASE=' + database + ';' \
            'Trusted_Connection=yes;' \
            'charset=' + charset

        # 'UID=nb-user;' \
        # 'PWD=nb-password;'

        params = urllib.parse.quote_plus(params)

        self.driver = driver
        self.server = server
        self.port = port
        self.database = database
        self.charset = charset

        self.sql_engine = sqlalchemy.create_engine('mssql+pyodbc:///?odbc_connect=%s' % params, fast_executemany=True)
        self.sql_connection = self.sql_engine.raw_connection()
        self.sql_cursor = self.sql_connection.cursor()
        self.sql_cursor.fast_executemany = True
