"""pytest 公共设施:把 src/ 加进 sys.path(项目 uv package=false,不安装自身)。"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
