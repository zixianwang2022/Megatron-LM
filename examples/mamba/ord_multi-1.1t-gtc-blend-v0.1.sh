#!/bin/bash

# multilingual datasets
CC2240_DATA_HOME="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nvllm-1.1t/data/tokens/non-english/CC-2022-40-Plus"
AR2240="${CC2240_DATA_HOME}/AR_shuf_text_document"
AZ2240="${CC2240_DATA_HOME}/AZ_shuf_text_document"
BG2240="${CC2240_DATA_HOME}/BG_shuf_text_document"
BN2240="${CC2240_DATA_HOME}/BN_shuf_text_document"
CA2240="${CC2240_DATA_HOME}/CA_shuf_text_document"
CS2240="${CC2240_DATA_HOME}/CS_shuf_text_document"
DA2240="${CC2240_DATA_HOME}/DA_shuf_text_document"
DE2240="${CC2240_DATA_HOME}/DE_shuf_text_document"
EL2240="${CC2240_DATA_HOME}/EL_shuf_text_document"
ES2240="${CC2240_DATA_HOME}/ES_shuf_text_document"
ET2240="${CC2240_DATA_HOME}/ET_shuf_text_document"
FA2240="${CC2240_DATA_HOME}/FA_shuf_text_document"
FI2240="${CC2240_DATA_HOME}/FI_shuf_text_document"
FR2240="${CC2240_DATA_HOME}/FR_shuf_text_document"
GL2240="${CC2240_DATA_HOME}/GL_shuf_text_document"
HE2240="${CC2240_DATA_HOME}/HE_shuf_text_document"
HI2240="${CC2240_DATA_HOME}/HI_shuf_text_document"
HR2240="${CC2240_DATA_HOME}/HR_shuf_text_document"
HU2240="${CC2240_DATA_HOME}/HU_shuf_text_document"
HY2240="${CC2240_DATA_HOME}/HY_shuf_text_document"
ID2240="${CC2240_DATA_HOME}/ID_shuf_text_document"
IS2240="${CC2240_DATA_HOME}/IS_shuf_text_document"
IT2240="${CC2240_DATA_HOME}/IT_shuf_text_document"
KA2240="${CC2240_DATA_HOME}/KA_shuf_text_document"
KK2240="${CC2240_DATA_HOME}/KK_shuf_text_document"
KN2240="${CC2240_DATA_HOME}/KN_shuf_text_document"
KO2240="${CC2240_DATA_HOME}/KO_shuf_text_document"
LT2240="${CC2240_DATA_HOME}/LT_shuf_text_document"
LV2240="${CC2240_DATA_HOME}/LV_shuf_text_document"
MK2240="${CC2240_DATA_HOME}/MK_shuf_text_document"
ML2240="${CC2240_DATA_HOME}/ML_shuf_text_document"
MR2240="${CC2240_DATA_HOME}/MR_shuf_text_document"
NE2240="${CC2240_DATA_HOME}/NE_shuf_text_document"
NL2240="${CC2240_DATA_HOME}/NL_shuf_text_document"
NO2240="${CC2240_DATA_HOME}/NO_shuf_text_document"
PL2240="${CC2240_DATA_HOME}/PL_shuf_text_document"
PT2240="${CC2240_DATA_HOME}/PT_shuf_text_document"
RO2240="${CC2240_DATA_HOME}/RO_shuf_text_document"
RU2240="${CC2240_DATA_HOME}/RU_shuf_text_document"
SK2240="${CC2240_DATA_HOME}/SK_shuf_text_document"
SL2240="${CC2240_DATA_HOME}/SL_shuf_text_document"
SQ2240="${CC2240_DATA_HOME}/SQ_shuf_text_document"
SR2240="${CC2240_DATA_HOME}/SR_shuf_text_document"
SV2240="${CC2240_DATA_HOME}/SV_shuf_text_document"
TA2240="${CC2240_DATA_HOME}/TA_shuf_text_document"
TE2240="${CC2240_DATA_HOME}/TE_shuf_text_document"
TR2240="${CC2240_DATA_HOME}/TR_shuf_text_document"
UK2240="${CC2240_DATA_HOME}/UK_shuf_text_document"
UR2240="${CC2240_DATA_HOME}/UR_shuf_text_document"
VI2240="${CC2240_DATA_HOME}/VI_shuf_text_document"

MC4_DATA_HOME="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nvllm-1.1t/data/tokens/non-english/mc4-ai2"
JAMC4="${MC4_DATA_HOME}/JA_shuf_text_document"
ZHMC4="${MC4_DATA_HOME}/ZH_shuf_text_document"

NMT_DATA_HOME="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nvllm-1.1t/data/tokens/non-english/nmt"
NMT="${NMT_DATA_HOME}/nmt_shuf_text_document"

#english datasets
ENG_DATA_HOME="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nvllm-1.1t/data/tokens/english"
B3="${ENG_DATA_HOME}/MTNLG/Books3_shuf_text_document"
OWT2="${ENG_DATA_HOME}/MTNLG/OpenWebText2_shuf_text_document"
SE="${ENG_DATA_HOME}/MTNLG/StackExchange_shuf_text_document"
PM="${ENG_DATA_HOME}/MTNLG/PubMedAbs_shuf_text_document"
WIK="${ENG_DATA_HOME}/MTNLG/Wikipedia_shuf_text_document"
GUT="${ENG_DATA_HOME}/MTNLG/Gutenberg_shuf_text_document"
BC2="${ENG_DATA_HOME}/MTNLG/BookCorpus2_shuf_text_document"
NIH="${ENG_DATA_HOME}/MTNLG/NIHExporter_shuf_text_document"
ARX="${ENG_DATA_HOME}/MTNLG/ArXiv_shuf_text_document"
ST="${ENG_DATA_HOME}/MTNLG/Stories_shuf_text_document"
CC202104="${ENG_DATA_HOME}/MTNLG/CC-2021-04_shuf_text_document"
PCC="${ENG_DATA_HOME}/MTNLG/Pile-CC_shuf_text_document"
#RN="${ENG_DATA_HOME}/MTNLG/RealNews_shuf_text_document"
BIGSC="${ENG_DATA_HOME}/BigScience/BigScience_shuf_text_document"
REDDIT="${ENG_DATA_HOME}/Reddit-Plus/Reddit_all_dialogue_shuf_text_document"
CCNEWS="${ENG_DATA_HOME}/CC-NEWS/CC-NEWS_shuf_text_document"
CC202050="${ENG_DATA_HOME}/CC-MAIN-2020-50/CC-MAIN-2020-50_shuf_text_document"
CC202240_0="${ENG_DATA_HOME}/CC-MAIN-2022-40/CC-MAIN-2022-40_00_shuf_text_document"
CC202240_1="${ENG_DATA_HOME}/CC-MAIN-2022-40/CC-MAIN-2022-40_01_shuf_text_document"
CC201935="${ENG_DATA_HOME}/CC-MAIN-2019-35/CC-MAIN-2019-35_shuf_text_document"
MC4="${ENG_DATA_HOME}/mc4-en_1T-url/mc4-en_shuf_text_document"

#code datasets
CODE_DATA_HOME="/lustre/fsw/portfolios/llmservice/projects/llmservice_nlp_fm/data/adlr-nlp-sharing/nvllm-1.1t/data/tokens/stack-subset"
ASMB="${CODE_DATA_HOME}/assembly_shuf_text_document"
CPLA="${CODE_DATA_HOME}/c_shuf_text_document"
CSHA="${CODE_DATA_HOME}/c-sharp_shuf_text_document"
CLIS="${CODE_DATA_HOME}/common_lisp_shuf_text_document"
CPPP="${CODE_DATA_HOME}/cPlusPlus_shuf_text_document"
CSSL="${CODE_DATA_HOME}/css_shuf_text_document"
CUDA="${CODE_DATA_HOME}/cuda_python_omniverse_shuf_text_document"
DART="${CODE_DATA_HOME}/dart_shuf_text_document"
DOCK="${CODE_DATA_HOME}/dockerfile_shuf_text_document"
FORT="${CODE_DATA_HOME}/fortran_shuf_text_document"
GOPL="${CODE_DATA_HOME}/go_shuf_text_document"
HASK="${CODE_DATA_HOME}/haskell_shuf_text_document"
HTML="${CODE_DATA_HOME}/html_shuf_text_document"
JAVA="${CODE_DATA_HOME}/java_shuf_text_document"
JASC="${CODE_DATA_HOME}/javascript_shuf_text_document"
JSON="${CODE_DATA_HOME}/json_shuf_text_document"
JULI="${CODE_DATA_HOME}/julia_shuf_text_document"
JUPY="${CODE_DATA_HOME}/jupyter-notebook_shuf_text_document"
LUAL="${CODE_DATA_HOME}/lua_shuf_text_document"
MAKE="${CODE_DATA_HOME}/makefile_shuf_text_document"
MARD="${CODE_DATA_HOME}/markdown_shuf_text_document"
PASC="${CODE_DATA_HOME}/pascal_shuf_text_document"
PERL="${CODE_DATA_HOME}/perl_shuf_text_document"
PHPL="${CODE_DATA_HOME}/php_shuf_text_document"
PYTH="${CODE_DATA_HOME}/python_shuf_text_document"
RSTL="${CODE_DATA_HOME}/rst_shuf_text_document"
RUBY="${CODE_DATA_HOME}/ruby_shuf_text_document"
RUST="${CODE_DATA_HOME}/rust_shuf_text_document"
SCAL="${CODE_DATA_HOME}/scala_shuf_text_document"
SHEL="${CODE_DATA_HOME}/shell_shuf_text_document"
SQLP="${CODE_DATA_HOME}/sql_shuf_text_document"
SWIF="${CODE_DATA_HOME}/swift_shuf_text_document"
TEXP="${CODE_DATA_HOME}/tex_shuf_text_document"
TYPE="${CODE_DATA_HOME}/typescript_shuf_text_document"
VISU="${CODE_DATA_HOME}/visual_basic_shuf_text_document"
XMLL="${CODE_DATA_HOME}/xml_shuf_text_document"
YAML="${CODE_DATA_HOME}/yaml_shuf_text_document"

DATA_BLEND="0.00005	${ASMB} \
0.01197	${CPLA} \
0.00697	${CSHA} \
0.00014	${CLIS} \
0.00919	${CPPP} \
0.00066	${CSSL} \
0.00022	${CUDA} \
0.00042	${DART} \
0.00001	${DOCK} \
0.00016	${FORT} \
0.00495	${GOPL} \
0.00022	${HASK} \
0.03512	${HTML} \
0.01444	${JAVA} \
0.02221	${JASC} \
0.00349	${JSON} \
0.00012	${JULI} \
0.00132	${JUPY} \
0.00032	${LUAL} \
0.00019	${MAKE} \
0.00069	${MARD} \
0.00010	${PASC} \
0.00025	${PERL} \
0.01262	${PHPL} \
0.01085	${PYTH} \
0.00012	${RSTL} \
0.00084	${RUBY} \
0.00168	${RUST} \
0.00065	${SCAL} \
0.00043	${SHEL} \
0.00097	${SQLP} \
0.00071	${SWIF} \
0.00020	${TEXP} \
0.00589	${TYPE} \
0.00011	${VISU} \
0.00117	${XMLL} \
0.00053	${YAML} \
0.01920	${B3} \
0.01602	${OWT2} \
0.00751	${SE} \
0.00324	${PM} \
0.00653	${WIK} \
0.00193	${GUT} \
0.00117	${BC2} \
0.00023	${NIH} \
0.01143	${ARX} \
0.00366	${ST} \
0.03992	${BIGSC} \
0.04768	${REDDIT} \
0.07199	${CCNEWS} \
0.02180	${PCC} \
0.07633	${CC202050} \
0.07644	${CC202240_0} \
0.07644	${CC202240_1} \
0.09414	${CC201935} \
0.03890	${CC202104} \
0.08544	${MC4} \
0.00117	${AR2240} \
0.00003	${AZ2240} \
0.00058	${BG2240} \
0.00010	${BN2240} \
0.00032	${CA2240} \
0.00157	${CS2240} \
0.00068	${DA2240} \
0.01892	${DE2240} \
0.00135	${EL2240} \
0.01768	${ES2240} \
0.00016	${ET2240} \
0.00139	${FA2240} \
0.00075	${FI2240} \
0.01660	${FR2240} \
0.00002	${GL2240} \
0.00020	${HE2240} \
0.00042	${HI2240} \
0.00039	${HR2240} \
0.00121	${HU2240} \
0.00002	${HY2240} \
0.00268	${ID2240} \
0.00003	${IS2240} \
0.00843	${IT2240} \
0.00003	${KA2240} \
0.00003	${KK2240} \
0.00001	${KN2240} \
0.00051	${KO2240} \
0.00024	${LT2240} \
0.00010	${LV2240} \
0.00003	${MK2240} \
0.00002	${ML2240} \
0.00003	${MR2240} \
0.00003	${NE2240} \
0.00429	${NL2240} \
0.00113	${NO2240} \
0.00389	${PL2240} \
0.00304	${PT2240} \
0.00139	${RO2240} \
0.01806	${RU2240} \
0.00041	${SK2240} \
0.00017	${SL2240} \
0.00005	${SQ2240} \
0.00016	${SR2240} \
0.00134	${SV2240} \
0.00009	${TA2240} \
0.00002	${TE2240} \
0.00136	${TR2240} \
0.00064	${UK2240} \
0.00003	${UR2240} \
0.00412	${VI2240} \
0.01234	${JAMC4} \
0.01017	${ZHMC4} \
0.01156	${NMT}"
