

# down tp weight and save as  yoloe26*-tp.pt


# 下载YOLOE26权重文件的函数
# 参数1：服务器主机名（可选，默认ultra6）
# 参数2：运行文件夹名称（可选，默认使用原固定值）
download_yoloe26_weight() {
    # 1. 定义参数（带默认值，传参则覆盖）
    local ultra_hostname=${1:-"ultra6"}  # 第1个参数：主机名，默认ultra6
    # 第2个参数：run_folder，默认使用原固定路径（保留兼容性）
    local run_folder=${2:-"26n_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[${ultra_hostname}]/"}
    local pro_folder=${3:-"yoloe26_tp/"}  # 可选参数，项目文件夹，默认yoloe26_tp
    local final_model_name=${4:-""}  # 可选参数，最终保存的模型名称，默认空
    
    # local dst_root="/Users/louis/yoloe26_weight/"  # 本地根目录（如需自定义可再加参数）
    # local dst_root= ${5:-}  # 本地根目录（如需自定义可再加参数）

    local remote_weight_path="/home/louis/ultra_louis_work/ultralytics/runs/${pro_folder}/${run_folder}/weights/best.pt"

    # 2. 拼接本地路径并创建目录
    dst_root="/home/louis/ultra_louis_work/ultralytics/runs/${pro_folder}/"
    local dst_dir="${dst_root}${run_folder}/weights/"
    local dst_path="${dst_dir}/best.pt"
    
    # 创建本地目录（-p确保父目录不存在时也能创建）
    mkdir -p "${dst_dir}"
    if [ $? -ne 0 ]; then
        echo "❌ 错误：创建本地目录失败 → ${dst_dir}"
        return 1
    fi


    # 检查文件是否已存在
    if [ -f "${dst_path}" ]; then
        echo "✅ 文件已存在，跳过下载：${dst_path}"
        du -h "${dst_path}"
        
        # # 如果提供了最终模型名称，复制到标准名称位置
        # if [ -n "${final_model_name}" ]; then
        #     mkdir -p "${dst_root}${pro_folder}"
        #     local standard_path="${dst_root}${pro_folder}${final_model_name}"
        #     if [ ! -f "${standard_path}" ]; then
        #         cp "${dst_path}" "${standard_path}"
        #         echo "✅ 已复制到标准路径：${standard_path}"
        #     fi
        # fi

        return 0
    fi

    # 3. 执行SCP下载（带进度显示）
    echo "🔄 正在从 ${ultra_hostname} 下载权重文件..."
    echo "   远程路径：louis@${ultra_hostname}:${remote_weight_path}"
    echo "   本地路径：${dst_path}"
    scp -p "louis@${ultra_hostname}:${remote_weight_path}" "${dst_path}"

    # 4. 验证下载结果
    if [ -f "${dst_path}" ]; then
        echo -e "\n✅ 成功！权重文件已保存到：\n${dst_path}"
        du -h "${dst_path}"
        
        # 如果提供了最终模型名称，复制到标准名称位置
        if [ -n "${final_model_name}" ]; then
            mkdir -p "${dst_root}${pro_folder}"
            local standard_path="${dst_root}${pro_folder}${final_model_name}"
            cp "${dst_path}" "${standard_path}"
            echo "✅ 已复制到标准路径：${standard_path}"
        fi
    else
        echo -e "\n❌ 失败！下载过程出错，未找到目标文件"
        return 1
    fi
}

# 



# # best tp weights
# download_yoloe26_weight ultra8 26n_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8] yoloe26_tp yoloe26n-tp.pt
# download_yoloe26_weight ultra8 26s_ptwobjv1_bs256_epo30_close2_engine_old_engine_data_tp[ultra8] yoloe26_tp yoloe26s-tp.pt
# download_yoloe26_weight ultra8 26m_ptwobjv1_bs256_epo25_close2_engine_old_engine_data_tp[ultra8] yoloe26_tp yoloe26m-tp.pt

# download_yoloe26_weight ultra6 26l_ptwobjv1_bs256_epo20_close2_engine_old_engine_data_tp[ultra6] yoloe26_tp yoloe26l-tp.pt
# download_yoloe26_weight ultra6 26x_ptwobjv1_bs256_epo15_close2_engine_old_engine_data_tp[ultra6] yoloe26_tp yoloe26x-tp.pt




# # best seg weights
# download_yoloe26_weight ultra2 26n-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2] yoloe26_seg yoloe26n_seg.pt
# download_yoloe26_weight ultra8 26s-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8] yoloe26_seg yoloe26s_seg.pt
# download_yoloe26_weight ultra2 26m-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra2] yoloe26_seg yoloe26m_seg.pt

# download_yoloe26_weight ultra6 26l-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6] yoloe26_seg yoloe26l_seg.pt
# download_yoloe26_weight ultra6 26x-seg_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6] yoloe26_seg yoloe26x_seg.pt





# # pf weights
# download_yoloe26_weight ultra8 26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8] yoloe26_pf yoloe26n_pf.pt
# download_yoloe26_weight ultra8 26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8] yoloe26_pf yoloe26m_pf.pt
# download_yoloe26_weight ultra6 26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6] yoloe26_pf yoloe26x_pf.pt
# download_yoloe26_weight ultra6 26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra6] yoloe26_pf yoloe26l_pf.pt
# download_yoloe26_weight ultra8 26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf[ultra8] yoloe26_pf yoloe26s_pf.pt


# # best vp weights

# download_yoloe26_weight ultra8 26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8] yoloe26_vp yoloe26n_vp.pt
# download_yoloe26_weight ultra8 26s_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8] yoloe26_vp yoloe26s_vp.pt
# download_yoloe26_weight ultra8 26m_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra8] yoloe26_vp yoloe26m_vp.pt
# download_yoloe26_weight ultra6 26l_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6] yoloe26_vp yoloe26l_vp.pt
# download_yoloe26_weight ultra6 26x_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_vp[ultra6] yoloe26_vp yoloe26x_vp.pt



# copy all files to 

download_yoloe26_weight ultra8 26n_ptwbest_tp_bs256_epo10_close2_engine_old_engine_data_pf2[ultra8] yoloe26_pf yoloe26n_pf.pt