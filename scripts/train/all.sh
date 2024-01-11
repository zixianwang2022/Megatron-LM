# lawrence mcafee

set -u

# ACCOUNT=adlr_nlp_llmnext
ACCOUNT=llmservice_dev_mcore
# ACCOUNT=llmservice_nlp_fm

# for USE_CORE in "0" "1"; do
#     for ADD_RETRIEVER in "0" "1"; do
#         for NWORKERS in "1" "2" "4" "8" "16" "32" "64" "128"; do
for USE_CORE in "0" "1"; do
    for ADD_RETRIEVER in "1"; do
        for NWORKERS in "8"; do
            echo "~~~~ launch c${USE_CORE}-r${ADD_RETRIEVER}-w${NWORKERS} ~~~~"
            sbatch \
                --export=USE_CORE="${USE_CORE}",ADD_RETRIEVER="${ADD_RETRIEVER}",NWORKERS="${NWORKERS}" \
                -A ${ACCOUNT} \
                --job-name=${ACCOUNT}-lmcafee:lmcafee_c${USE_CORE}-r${ADD_RETRIEVER}-w${NWORKERS} \
                ./single.sh
        done
    done
done

# eof
