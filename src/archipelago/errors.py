class ClusteringError(Exception):
    pass


# The logs are missing something the module needs, or carry an unsupported schema version.
class SchemaError(ClusteringError):
    pass


class UnsupportedEncoding(ClusteringError):
    pass
