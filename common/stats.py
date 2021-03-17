import tracemalloc
from time import time

from common.oracle import Oracle


class StatCollector:
    def __init__(self):
        self.__start_time = None
        self.__traces = False
        self.elapsed_time = 0.0
        self.fcalls = 0
        self.gcalls = 0
        self.hcalls = 0
        self.memory = 0
        self.arithm = 0
        self.extra = {}

    def clear(self):
        self.__start_time = None
        self.elapsed_time = 0.0
        self.fcalls = 0
        self.gcalls = 0
        self.hcalls = 0
        self.memory = 0
        self.arithm = 0
        self.extra = {}

    def start_clock(self):
        assert self.__start_time is None
        self.__start_time = time()

    def stop_clock(self):
        assert self.__start_time is not None
        self.elapsed_time += time() - self.__start_time
        self.__start_time = None

    def start_trace(self):
        assert not self.__traces
        self.__traces = True
        tracemalloc.start()

    def stop_trace(self):
        assert self.__traces
        res = tracemalloc.take_snapshot()
        stats = res.statistics(cumulative=True, key_type='filename')
        for stat in stats:
            self.memory += stat.size
        tracemalloc.stop()
        self.__traces = False

    def start(self):
        self.start_clock()
        self.start_trace()

    def stop(self):
        self.stop_clock()
        self.stop_trace()

    def report(self, op, cnt):
        if op in ['+', '-', '*', '/']:
            self.arithm += cnt
        else:
            self.extra[op] = self.extra.get(op, 0) + cnt

    def assimilate(self, oracle: Oracle):
        self.fcalls += oracle.calls
        self.gcalls += oracle.gcalls
        self.hcalls += oracle.hcalls
