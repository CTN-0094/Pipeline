# Constants

**File:** `src/constants.py`

---

## `EndpointType`

An enum that defines the three supported outcome endpoint types. Used throughout the pipeline to select the correct model class and validation rules.

```python
from src.constants import EndpointType
```

| Member | Value | Meaning |
|:---|:---|:---|
| `EndpointType.LOGICAL` | `"logical"` | Binary classification (0/1 outcome) |
| `EndpointType.INTEGER` | `"integer"` | Count regression (non-negative integer outcome) |
| `EndpointType.SURVIVAL` | `"survival"` | Time-to-event (duration + event indicator) |

### Usage

```python
from src.constants import EndpointType

# Construct an outcome config
outcome = {
    "endpointType": EndpointType.LOGICAL,
    "columnsToUse": ["ctn0094_relapse_event"]
}

# Compare
if outcome["endpointType"] == EndpointType.LOGICAL:
    print("Binary classification task")

# Instantiate from string
endpoint = EndpointType("logical")  # → EndpointType.LOGICAL
```

!!! note
    Passing an invalid string to `EndpointType()` raises a `ValueError`. All pipeline entry points validate the endpoint type before any data processing begins.
