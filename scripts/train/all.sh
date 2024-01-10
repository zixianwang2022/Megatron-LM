# lawrence mcafee

# ACCOUNT=adlr_nlp_llmnext
ACCOUNT=llmservice_dev_mcore
# ACCOUNT=llmservice_nlp_fm

# for USE_CORE in "0" "1"; do
#     for ADD_RETRIEVER in "0" "1"; do
#         for TP in "4"; do
for USE_CORE in "1"; do
    for ADD_RETRIEVER in "1"; do
        for TP in "1"; do
            echo "~~~~ launch c${USE_CORE}-r${ADD_RETRIEVER}-t${TP} ~~~~"
            sbatch \
                --export=USE_CORE="${USE_CORE}",ADD_RETRIEVER="${ADD_RETRIEVER}",TP="${TP}" \
                -A ${ACCOUNT} \
                --job-name=${ACCOUNT}-lmcafee:lmcafee_c${USE_CORE}-r${ADD_RETRIEVER}-t${TP} \
                ./single.sh
        done
    done
done

# eof
