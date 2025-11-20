# heart_rate.py

from processor import Processor

class HeartRate(Processor):
    def __init__(self, frequency: int):
        super().__init__(frequency)
        self.__heart_rate = 0

    def add_sample(self, ir_sample: int):
        """
        Processor._add_sample expected (ir, red)
        but MAX30102 IR-only HR uses red_sample=ir or 0
        => to keep Processor happy, we duplicate IR
        """
        self._add_sample(ir_sample, ir_sample)

        self.__heart_rate = self.__calculate_heart_rate()

    def get(self) -> int:
        if not self._presence():   # Processor presence check
            return -1
        return self.__heart_rate

    def __calculate_heart_rate(self) -> int:
        peaks = self._peaks()      # Processor peak detection

        if len(peaks) < 2:
            return self.__heart_rate

        # Calculate intervals (sec between peaks)
        intervals = [peaks[i][0] - peaks[i-1][0] for i in range(1, len(peaks))]

        # Sliding window
        self._intervals.extend(intervals)
        self._intervals = self._intervals[-self._processed_window_size:]

        if len(self._intervals) > 1:
            avg_interval = sum(self._intervals[-self._MOVING_AVERAGE_WINDOW:]) / self._MOVING_AVERAGE_WINDOW
            return int(60 / avg_interval)

        return self.__heart_rate
