"""Conservative empirical admission; no invented hardware bandwidth metrics."""

from dataclasses import dataclass

from .policy import CapacityPoint


def context_bucket(tokens: int) -> int:
    return 1 << max(10, (max(1, tokens) - 1).bit_length())


@dataclass
class Estimate:
    aggregate_tps: float
    samples: int


class CapacityModel:
    def __init__(self, points: tuple[CapacityPoint, ...] = (), margin=0.15):
        self.margin = margin
        self.points = {
            (p.configuration, p.context_bucket, p.concurrency): Estimate(
                p.aggregate_tps, p.samples
            )
            for p in points
        }

    def observe(self, configuration, bucket, concurrency, aggregate_tps):
        if concurrency < 1 or aggregate_tps <= 0:
            return
        key = configuration, bucket, concurrency
        previous = self.points.get(key)
        self.points[key] = Estimate(
            0.2 * aggregate_tps + 0.8 * previous.aggregate_tps
            if previous
            else aggregate_tps,
            previous.samples + 1 if previous else 1,
        )

    def predict(self, configuration, bucket, concurrency):
        point = self.points.get((configuration, bucket, concurrency))
        if point is None:
            return None
        return point.aggregate_tps * (1 - self.margin) / concurrency

    def allows(self, candidate, active):
        streams = [*active, candidate]
        # Different models/configurations contend for the same physical GPU.
        # No independent fictitious capacity pools on a single appliance.
        if any(s.configuration != candidate.configuration for s in active):
            return False, "mixed_configuration_unmeasured"
        bucket = max(s.bucket for s in streams)
        predicted = self.predict(candidate.configuration, bucket, len(streams))
        if predicted is None:
            # Best-effort singleton is the only uncalibrated exploration mode.
            return (not active and candidate.policy.min_tps is None), "uncalibrated"
        for stream in streams:
            floor = stream.policy.min_tps
            if floor is not None and predicted < floor:
                return False, "predicted_floor_breach"
            snapshot = stream.metrics.snapshot()
            rolling = snapshot["rolling_5s_tps"]
            # A five-second evidence window avoids reacting to one slow step.
            if (
                floor
                and rolling is not None
                and stream.metrics.first_token is not None
                and stream.metrics.clock() - stream.metrics.first_token >= 5
                and rolling < floor
            ):
                return False, "observed_floor_breach"
        return True, "calibrated"
