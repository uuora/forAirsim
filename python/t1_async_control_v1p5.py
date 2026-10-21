"""Receipt-only correction; all V1P4 control and stopping limits remain."""
from t1_async_control_v1p4 import AsyncControlV1P4
from t1_zero_hold_proof_v1 import stable_interval_proof


class AsyncControlV1P5(AsyncControlV1P4):
    revision="STRICT_STOP_STABLE_INTERVAL_ZERO_HOLD_RECEIPTS_V1P5"

    def _consume(self,flight,policy,state,proposal):
        result=super()._consume(flight,policy,state,proposal)
        if result.get("zero_held_stop_proof") is not None:
            result["zero_held_stop_proof"]=stable_interval_proof(result["zero_held_stop_proof"])
        return result
