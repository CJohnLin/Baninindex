class ActionEnum:
    HEAVY_SHORT = "📉 重倉做空"
    SHORT = "🔽 輕倉做空"
    HOLD = "👀 觀望不動作"
    LONG = "🔼 輕倉做多"
    HEAVY_LONG = "🚀 重倉做多"

def get_action_weight(action_name):
    mapping = {
        ActionEnum.HEAVY_SHORT: -1.0,
        ActionEnum.SHORT: -0.5,
        ActionEnum.HOLD: 0.0,
        ActionEnum.LONG: 0.5,
        ActionEnum.HEAVY_LONG: 1.0,
    }
    return mapping.get(action_name, 0.0)

def decide_action(bert_score: float, emotion: str) -> str:
    """
    根據 AI 的反轉機率分數與社群情境，代理人自主決定投資動作。
    """
    # 判斷市場反轉預期方向
    expected_market_dir = 0 # 0=觀望, -1=跌, 1=漲
    if "跌" in emotion or "高機率下跌" in emotion: # 原PO看多或被套死抱 -> 反向看跌
        expected_market_dir = -1
    elif "漲" in emotion or "反彈" in emotion:     # 原PO看空或停損 -> 反向看漲
        expected_market_dir = 1
        
    if expected_market_dir == 0:
        return ActionEnum.HOLD
        
    # 評估信心指數 (bert_score) 以決定下注大小
    if bert_score > 0.85:
        return ActionEnum.HEAVY_LONG if expected_market_dir == 1 else ActionEnum.HEAVY_SHORT
    elif bert_score > 0.65:
        return ActionEnum.LONG if expected_market_dir == 1 else ActionEnum.SHORT
    else:
        return ActionEnum.HOLD
