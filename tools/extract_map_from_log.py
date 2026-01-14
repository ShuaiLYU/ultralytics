import ultralytics,os
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)



import ultralytics
import os
import re

# 设置ultralytics工作空间
workspace = os.path.dirname(os.path.dirname(os.path.abspath(ultralytics.__file__)))
os.chdir(workspace)
print("set workspace:", workspace)

def extract_ap_ar_lines(input_log_path, suffix="_extract"):
    """
    从log文件中提取COCO评估相关的所有关键行：
    1. Evaluate annotation type（含*bbox*/*segm*）
    2. COCOeval_opt相关过程行
    3. AP/AR指标行
    输出文件路径自动基于输入路径生成：原文件名 + 后缀 + .log
    
    Args:
        input_log_path: 原始log文件路径
        suffix: 输出文件的后缀（默认是 "_extract"）
    """
    # 解析输入文件路径，生成输出路径
    input_dir = os.path.dirname(input_log_path)
    input_filename = os.path.basename(input_log_path)
    name_without_ext, ext = os.path.splitext(input_filename)
    output_log_path = os.path.join(input_dir, f"{name_without_ext}{suffix}{ext}")

    # 定义匹配规则（适配所有目标行，包括特殊符号、空格、大小写）
    # 覆盖：
    # 1. Evaluate annotation type *xxx* （带星号的标注类型）
    # 2. COCOeval_opt.evaluate()/accumulate() 过程行
    # 3. DONE (t=xx.xxs) 耗时行
    # 4. Accumulating evaluation results... 结果汇总行
    # 5. AP/AR指标行
    pattern = re.compile(
        r'^\s*(Evaluate annotation type \*[\w]+\*|'  # 匹配 *bbox*/*segm* 类型
        r'COCOeval_opt\.(evaluate|accumulate)\(\) finished...|'  # COCOeval过程行
        r'DONE \(t=\d+\.\d+s\)\.|'  # 耗时行
        r'Accumulating evaluation results...|'  # 结果汇总行
        r'Average (Precision|Recall)\s+\((AP|AR)\)\s+@\[.+\] = .+)$'  # AP/AR指标行
    )
    
    # 存储提取的行
    extracted_lines = []
    
    # 读取并匹配log文件
    try:
        with open(input_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped = line.rstrip('\n')  # 保留原始格式（除换行符）
                # 严格匹配目标行
                if pattern.match(line_stripped):
                    extracted_lines.append(line_stripped)
        
        # 写入输出文件
        with open(output_log_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(extracted_lines) + '\n')
        
        print(f"✅ 成功提取 {len(extracted_lines)} 行COCO评估相关内容")
        print(f"📄 结果已保存至: {output_log_path}")
        
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件 {input_log_path}")
    except Exception as e:
        print(f"❌ 处理出错：{str(e)}")

# -------------------------- 配置运行参数 --------------------------
if __name__ == "__main__":
    # 你的原始log文件路径
    INPUT_LOG_FILE = "/home/louis/ultra_louis_work/ultralytics/tp_only_val_yoloe26_seg_26s_20260112_135646.log"
    
    #tp_only_val_yoloe26_seg_26x_20260112_135710.log
    INPUT_LOG_FILE="/home/louis/ultra_louis_work/ultralytics/tp_only_val_yoloe26_seg_26x_20260112_135710.log"

    INPUT_LOG_FILE="/home/louis/ultra_louis_work/ultralytics/tp_only_val_yoloe26_seg_26m_20260112_135651.log"

    INPUT_LOG_FILE="/home/louis/ultra_louis_work/ultralytics/tp_only_val_yoloe26_seg_26n_20260112_191927.log"

    INPUT_LOG_FILE="/home/louis/ultra_louis_work/ultralytics/tp_only_val_yoloe26_seg_26l_20260112_193119.log"

    INPUT_LOG_FILE="./tp_vp_26n_bbox20260113_102043.log"
    # INPUT_LOG_FILE="./tp_vp_26s_bbox20260113_102200.log"
    # INPUT_LOG_FILE="./tp_vp_26m_bbox20260113_102208.log"
    # INPUT_LOG_FILE="./tp_vp_26l_bbox20260113_102214.log"
    # INPUT_LOG_FILE="./tp_vp_26x_bbox20260113_102222.log"


    # 调用函数（可自定义后缀，如 suffix="_coco_eval"）
    extract_ap_ar_lines(INPUT_LOG_FILE, suffix="_extract")