"""生成《项目文档.pdf》：python docs/build_doc.py [输出路径]
数字全部从三个项目的 results/ 读取，重新跑实验后重新执行本脚本即可更新文档。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import content_common  # noqa: E402
import content_p1  # noqa: E402
import content_p2  # noqa: E402
import content_p3  # noqa: E402
from pdfkit import Doc, build  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "项目文档.pdf")
    d = Doc()
    content_common.overview(d)
    content_p1.content(d)
    content_p2.content(d)
    content_p3.content(d)
    content_common.appendix(d)
    build(d, out, content_common.cover(), "大模型算法岗三项目 · 项目文档")
    print("->", out)


if __name__ == "__main__":
    main()
