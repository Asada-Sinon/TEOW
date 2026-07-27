"""pytest 公共设施:把 src/ 加进 sys.path(项目 uv package=false,不安装自身)。"""

import pathlib
import sys

_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "explorations"))  # slow test 测 explorations 训练管线(未进 src)
