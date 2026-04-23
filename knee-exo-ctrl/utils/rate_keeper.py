from __future__ import annotations

import time
from typing import Optional, Tuple


class RateKeeper:
    def __init__(self, hz: float, catchup_cycles: int = 3, spin_ns: int = 50_000):
        self.period_ns = int(1e9 / float(hz))
        self.next_ns: Optional[int] = None
        self.tick = 0
        self.catchup_cycles = int(catchup_cycles)
        self.spin_ns = int(spin_ns)

    def start(self) -> None:
        self.next_ns = time.perf_counter_ns()

    def wait(self) -> Tuple[float, float, int]:
        assert self.next_ns is not None
        self.next_ns += self.period_ns
        while True:
            now_ns = time.perf_counter_ns()
            dt_ns = self.next_ns - now_ns
            if dt_ns <= 0:
                late_cycles = (-dt_ns) // self.period_ns
                if late_cycles > self.catchup_cycles:
                    self.next_ns += late_cycles * self.period_ns
                    self.tick += int(late_cycles)
                break
            if dt_ns > self.spin_ns:
                time.sleep((dt_ns - self.spin_ns) / 1e9)
            else:
                while (self.next_ns - time.perf_counter_ns()) > 0:
                    pass
                break
        sched_s = self.tick * (self.period_ns / 1e9)
        overrun_s = max(0.0, -dt_ns / 1e9)
        self.tick += 1
        return overrun_s, sched_s, self.tick - 1
