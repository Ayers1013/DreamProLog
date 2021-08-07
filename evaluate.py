

class Judge:
  def __init__(self, agent, logger, eval_ds):
    self._agent=agent
    self._logger=logger
    self._ds=eval_ds

  def eval_dynamic(self):
    pass