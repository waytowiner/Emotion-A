"""生成板块 ↔ ETF 映射 config/etf_map.yaml

以 data/cache/board_list.parquet 的 100 个板块为键：
1) MANUAL 表内的板块用业内公认的主力 ETF（写死）；
2) 其余按关键词在 ETF 清单中自动匹配（板块名与 ETF 名互含/别名命中），
   候选按成交额排序取前 1~2 只；
3) 无对应的标 etf: null。

用法：python scripts/build_etf_map.py   （需先跑 scripts/fetch_etf_list.py）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "config" / "etf_map.yaml"

# ---------------- 人工校正：业内公认主力 ETF ----------------
# board_code -> [etf_code, ...]（名称以 etf_list 为准）
MANUAL: dict[str, list[str]] = {
    "BK1036": ["512480", "159995"],  # 半导体 -> 半导体ETF / 芯片ETF
    "BK1331": ["159995"],            # 数字芯片设计 -> 芯片ETF
    "BK1330": ["159995"],            # 模拟芯片设计 -> 芯片ETF
    "BK1201": ["515260"],            # 电子 -> 电子ETF
    "BK1207": ["512720"],            # 计算机 -> 计算机ETF
    "BK1215": ["515880"],            # 通信 -> 通信ETF
    "BK0486": ["512980"],            # 传媒 -> 传媒ETF
    "BK1046": ["159869"],            # 游戏Ⅱ -> 游戏ETF
    "BK1301": ["159869"],            # 游戏Ⅲ -> 游戏ETF
    "BK1408": ["562500"],            # 机器人 -> 机器人ETF
    "BK1617": ["518880"],            # 黄金 -> 黄金ETF（商品）
    "BK0732": ["518880"],            # 贵金属 -> 黄金ETF（商品）
    "BK1233": ["512660"],            # 军工电子Ⅱ -> 军工ETF
    "BK1386": ["512660"],            # 军工电子Ⅲ -> 军工ETF
    "BK1232": ["512660"],            # 航天装备Ⅱ -> 军工ETF
    "BK1385": ["512660"],            # 航天装备Ⅲ -> 军工ETF
    "BK1602": ["512170"],            # 诊断服务 -> 医疗ETF
    "BK1599": ["512170"],            # 其他医疗服务 -> 医疗ETF
    "BK1325": ["512480"],            # 半导体材料 -> 半导体ETF
    "BK1326": ["159516"],            # 半导体设备 -> 半导体设备ETF
    # 电子上游元器件/材料/设备类：统一挂宽基电子ETF（避免误配消费电子）
    "BK1340": ["515260"],            # 印制电路板 -> 电子ETF
    "BK0459": ["515260"],            # 元件 -> 电子ETF
    "BK1334": ["515260"],            # 光学元件 -> 电子ETF
    "BK1339": ["515260"],            # 被动元件 -> 电子ETF
    "BK1333": ["515260"],            # LED -> 电子ETF
    "BK1038": ["515260"],            # 光学光电子 -> 电子ETF
    "BK1332": ["515260"],            # 电子化学品Ⅲ -> 电子ETF
    "BK1039": ["515260"],            # 电子化学品Ⅱ -> 电子ETF
    "BK1336": ["515260"],            # 其他电子Ⅲ -> 电子ETF
    "BK1223": ["515260"],            # 其他电子Ⅱ -> 电子ETF
}

# ---------------- 同花顺板块池人工表：板块名 -> [etf_code, ...] ----------------
MANUAL_THS: dict[str, list[str]] = {
    "白酒": ["512690"],              # 酒ETF
    "饮料制造": ["515170"],          # 食品饮料ETF
    "食品加工制造": ["515170"],
    "证券": ["512880"],              # 证券ETF
    "保险": ["512070"],              # 证券保险ETF
    "多元金融": ["512070"],
    "银行": ["512800"],              # 银行ETF
    "化学制药": ["512010"],          # 医药ETF
    "生物制品": ["512010"],
    "医药商业": ["512010"],
    "中药": ["560080"],              # 中药ETF
    "医疗服务": ["512170"],          # 医疗ETF
    "医疗器械": ["512170"],
    "光伏设备": ["515790"],          # 光伏ETF
    "电池": ["159755"],              # 电池ETF
    "风电设备": ["515790"],
    "汽车整车": ["516110"],          # 汽车ETF
    "汽车零部件": ["516110"],
    "半导体": ["512480", "159995"],  # 半导体ETF / 芯片ETF
    "军工电子": ["512660"],          # 军工ETF
    "军工装备": ["512660"],
    "计算机设备": ["512720"],        # 计算机ETF
    "IT服务": ["512720"],
    "软件开发": ["159852"],          # 软件ETF
    "通信设备": ["515880"],          # 通信ETF
    "通信服务": ["515880"],
    "文化传媒": ["512980"],          # 传媒ETF
    "游戏": ["159869"],              # 游戏ETF
    "影视院线": ["512980"],
    "电力": ["159611"],              # 电力ETF
    "电网设备": ["159611"],
    "煤炭开采加工": ["515220"],      # 煤炭ETF
    "钢铁": ["515210"],              # 钢铁ETF
    "工业金属": ["512400"],          # 有色金属ETF
    "小金属": ["512400"],
    "金属新材料": ["512400"],
    "能源金属": ["512400"],
    "贵金属": ["518880"],            # 黄金ETF（商品）
    "房地产": ["512200"],            # 房地产ETF
    "白色家电": ["159996"],          # 家电ETF
    "厨卫电器": ["159996"],
    "小家电": ["159996"],
    "黑色家电": ["159996"],
    "旅游及酒店": ["159766"],        # 旅游ETF
    "农产品加工": ["159825"],        # 农业ETF
    "种植业与林业": ["159825"],
    "养殖业": ["159865"],            # 养殖ETF
    "建筑材料": ["159745"],          # 建材ETF
    "建筑装饰": ["516970"],          # 基建ETF
    "物流": ["516910"],              # 物流ETF
    "化学制品": ["159870"],          # 化工ETF
    "化学原料": ["159870"],
    "农化制品": ["159870"],
    "教育": ["513360"],              # 教育ETF
    "消费电子": ["159732"],          # 消费电子ETF
    "其他电子": ["515260"],          # 电子ETF
    "电子化学品": ["515260"],
    "光学光电子": ["515260"],
    "元件": ["515260"],
    "自动化设备": ["562500"],        # 机器人ETF
    "机器人": ["562500"],
    "工程机械": ["516960"],          # 工程机械ETF
    "专用设备": ["516960"],
    "通用设备": ["516960"],
}

# ---------------- 自动匹配别名：板块名 -> ETF 名关键词（按优先级） ----------------
ALIASES: dict[str, list[str]] = {
    "半导体材料": ["半导体材料", "半导体"],
    "半导体设备": ["半导体设备", "半导体"],
    "集成电路封测": ["半导体", "芯片"],
    "集成电路制造": ["半导体", "芯片"],
    "分立器件": ["半导体", "芯片"],
    "印制电路板": ["电子"],
    "元件": ["电子"],
    "被动元件": ["电子"],
    "光学元件": ["电子"],
    "光学光电子": ["电子"],
    "LED": ["电子"],
    "其他电子": ["电子"],
    "电子化学品": ["电子"],
    "消费电子零部件及组装": ["消费电子"],
    "计算机设备": ["计算机"],
    "其他计算机设备": ["计算机"],
    "IT服务": ["计算机", "软件"],
    "软件开发": ["软件", "计算机"],
    "横向通用软件": ["软件", "计算机"],
    "垂直应用软件": ["软件", "计算机"],
    "通信设备": ["通信"],
    "通信服务": ["通信"],
    "通信网络设备及器件": ["通信"],
    "通信线缆及配套": ["通信"],
    "其他通信设备": ["通信"],
    "通信工程及服务": ["通信"],
    "通信应用增值服务": ["通信"],
    "数字媒体": ["传媒"],
    "文字媒体": ["传媒"],
    "其他数字媒体": ["传媒"],
    "图片媒体": ["传媒"],
    "门户网站": ["传媒"],
    "电视广播": ["传媒", "广播电视"],
    "广告营销": ["传媒"],
    "营销代理": ["传媒"],
    "影视院线": ["影视", "传媒"],
    "影视动漫制作": ["影视", "传媒"],
    "教育": ["教育"],
    "学历教育": ["教育"],
    "教育运营及其他": ["教育"],
    "自动化设备": ["机器人", "智能制造"],
    "其他自动化设备": ["机器人", "智能制造"],
    "工控设备": ["机器人", "智能制造"],
    "机械设备": ["机械"],
    "通用设备": ["机械"],
    "其他通用设备": ["机械"],
    "机床工具": ["机床", "工业母机", "机械"],
    "环保设备": ["环保", "碳中和"],
    "综合环境治理": ["环保", "碳中和"],
    "小金属": ["稀有金属", "有色金属"],
    "其他小金属": ["稀有金属", "有色金属"],
    "钨": ["稀有金属", "有色金属"],
    "金属新材料": ["新材料", "稀有金属"],
    "其他金属新材料": ["新材料", "稀有金属"],
    "塑料": ["化工", "新材料"],
    "改性塑料": ["化工", "新材料"],
    "合成树脂": ["化工", "新材料"],
    "膜材料": ["新材料", "化工"],
    "胶黏剂及胶带": ["化工", "新材料"],
    "激光设备": ["激光"],
    "装修装饰": ["建筑", "基建"],
    "园林工程": ["基建", "建筑"],
    "其他专业工程": ["基建", "建筑"],
}

# ETF 名中的排除词（债/货/海外等不匹配行业板块的品种）
EXCLUDE = ["债", "货币", "添利", "现金", "纳斯达克", "纳指", "标普", "道琼斯",
           "恒生", "港股", "H股", "中概", "日经", "德国", "法国", "沙特",
           "油气", "原油", "豆粕", "白银", "有色期货",
           "中韩",      # 跨境 QDII，非 A 股行业
           "科创",      # 科创板窄基，口径比行业板块窄
           "能源化工", "期货"]  # 商品期货 ETF，非股票行业
# 教育板特例：513360 为 QDII 全球中国教育，波动逻辑不同，仍允许（散户常问）
ALLOW_QDII = {"教育": ["513360"]}

NUM_SUFFIX = re.compile(r"[ⅡⅢⅣIV]+$")


def norm_board(name: str) -> str:
    n = NUM_SUFFIX.sub("", name).strip()
    return n


def auto_match(bname: str, etfs: pd.DataFrame) -> tuple[list[dict], str]:
    """返回 (候选列表, 命中关键词)。"""
    base = norm_board(bname)
    keywords = ALIASES.get(bname) or ALIASES.get(base) or ([base] if len(base) >= 2 else [])
    for kw in keywords:
        cand = etfs[etfs["name"].str.contains(kw, na=False)]
        cand = cand[~cand["name"].str.contains("|".join(EXCLUDE), na=False)]
        if cand.empty:
            continue
        cand = cand.sort_values("amount", ascending=False)
        picks = cand.head(2)
        return ([{"code": r.code, "name": r.name} for r in picks.itertuples()], kw)
    return [], ""


def main() -> None:
    source = "eastmoney"
    if "--source" in sys.argv:
        source = sys.argv[sys.argv.index("--source") + 1]
    board_file = "board_list_ths.parquet" if source == "ths" else "board_list.parquet"
    boards = pd.read_parquet(CACHE / board_file)
    etf_path = CACHE / "etf_list.parquet"
    if not etf_path.exists():
        print("缺少 data/cache/etf_list.parquet，请先运行 scripts/fetch_etf_list.py")
        sys.exit(1)
    etfs = pd.read_parquet(etf_path)
    etfs["code"] = etfs["code"].astype(str).str.zfill(6)
    by_code = {r.code: r.name for r in etfs.itertuples()}

    lines = [
        "# 板块 ↔ ETF 映射（scripts/build_etf_map.py 生成，可手工编辑后重跑 dump）",
        "# etf: null 表示无直接对应 ETF；注释注明匹配方式（manual/auto:关键词）",
    ]
    n_have = 0
    for r in boards.itertuples():
        code, bname = r.board_code, r.board_name
        # 同花顺口径人工表按板块名键控；东财口径按板块代码
        manual_codes = MANUAL_THS.get(bname) if source == "ths" else MANUAL.get(code)
        lines.append(f"{code}:")
        lines.append(f"  name: {bname}")
        if manual_codes:
            picks, missing = [], []
            for ec in manual_codes:
                if ec in by_code:
                    picks.append({"code": ec, "name": by_code[ec]})
                else:
                    missing.append(ec)
            if missing:
                print(f"[warn] {bname} 手动 ETF 不在清单内：{missing}")
            if picks:
                n_have += 1
                lines.append("  etf:")
                for p in picks:
                    lines.append(f'    - {{code: "{p["code"]}", name: "{p["name"]}"}}  # manual')
            else:
                lines.append("  etf: null  # 手动指定的 ETF 未在清单中找到")
            continue
        picks, kw = auto_match(bname, etfs)
        if picks:
            n_have += 1
            lines.append("  etf:")
            for p in picks:
                lines.append(f'    - {{code: "{p["code"]}", name: "{p["name"]}"}}  # auto:{kw}')
        else:
            lines.append("  etf: null  # 无直接ETF")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = len(boards)
    print(f"\n覆盖率：{n_have}/{total} 个板块有 ETF（{n_have / total:.0%}）-> {OUT}")

    # 抽查：常见板块
    print("\n抽查：")
    import yaml
    m = yaml.safe_load(OUT.read_text(encoding="utf-8"))
    spot = ["半导体", "酒", "证券", "军工", "医药", "光伏", "新能源车", "银行", "传媒", "游戏"]
    name2code = {v["name"]: k for k, v in m.items()}
    for s in spot:
        hit = [(k, v) for k, v in m.items() if s in v["name"]]
        if not hit:
            print(f"  {s}: 不在当前板块池内")
        for k, v in hit:
            etfs_s = ", ".join(f'{e["code"]}{e["name"]}' for e in (v["etf"] or [])) or "无直接ETF"
            print(f"  {s} -> {v['name']}({k}): {etfs_s}")


if __name__ == "__main__":
    main()
