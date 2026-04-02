# check_imports.py
import ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

# Check available imports
try:
    from ultralytics.utils import colorstr
    print("✓ colorstr available")
except ImportError as e:
    print(f"✗ colorstr: {e}")

try:
    from ultralytics.utils.ops import xywh2xyxy
    print("✓ xywh2xyxy available")
except ImportError as e:
    print(f"✗ xywh2xyxy: {e}")

try:
    from ultralytics.utils.loss import BboxLoss
    print("✓ BboxLoss available")
except ImportError as e:
    print(f"✗ BboxLoss: {e}")

try:
    from ultralytics.utils.tal import TaskAlignedAssigner
    print("✓ TaskAlignedAssigner available")
except ImportError as e:
    print(f"✗ TaskAlignedAssigner: {e}")

# Check model imports
try:
    from ultralytics.nn.tasks import DetectionModel
    print("✓ DetectionModel available")
except ImportError as e:
    print(f"✗ DetectionModel: {e}")