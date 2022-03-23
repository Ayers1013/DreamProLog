'''
AnalyticsModule provide functions to log useful metrics in sublayers.
Warning: can only be used if the subclass also inherits from ConfiguredModule.
'''

class AnalyticsModule:
    def __init__(self) -> None:
        super().__init__()
        self.__analytics = {}

    def log_analytics(self, name, value):
        self.__analytics[name] = value

    @property
    def analytics(self):
        collected_logs = {}
        for node in self:
            if isinstance(node, AnalyticsModule):
                name = node._nested_name
                collected_logs.update({name + '/' + k: v for k,v in node.__analytics.items()})

        return collected_logs
