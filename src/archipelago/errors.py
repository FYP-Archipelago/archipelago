class ClusteringError(Exception):
    pass


# The logs are missing something the module needs.
class SchemaError(ClusteringError):
    pass


class UnsupportedEncoding(ClusteringError):
    pass
