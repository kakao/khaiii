#!/usr/bin/env bash
set -e -u


#############
# functions #
#############
function print_usage() {
    local msg=$1
    echo "Usage: $(basename $0) [options]"
    echo "Options:"
    echo "  -h, --help        show this help message and exit"
    echo "  -i FILE           input file"
    echo "  -c DIR            corpus dir"
    echo "  --rsc-src=DIR     <default: ../../rsc/src>"
    echo "  --lib-path=FILE   <default: ../../build/lib/libkhaiii.so>"
    echo "  --rsc-dir=DIR     <default: ../../build/share/khaiii>"
    echo "  --num-mapper=NUM  <default: 1000>"
    if [ -z "${msg}" ]; then
        exit 0
    else
        echo
        echo "${msg}"
        exit 1
    fi
}


function abspath() {
    python3 -c "import os, sys; print(os.path.abspath(sys.argv[1]))" $1
}


function parse_args() {
    INPUT_FILE=""
    CORPUS_DIR=""
    LIB_PATH=""
    RSC_DIR=""
    RSC_SRC=""
    NUM_MAPPER=""

    while [[ $# -ge 1 ]]; do
        case $1 in
            -h|--help)
                print_usage ""
                ;;
            -i)
                INPUT_FILE="$2"
                shift
                ;;
            -c)
                CORPUS_DIR="$2"
                shift
                ;;
            --rsc-src)
                RSC_SRC="$2"
                shift
                ;;
            --lib-path)
                LIB_PATH="$2"
                shift
                ;;
            --rsc-dir)
                RSC_DIR="$2"
                shift
                ;;
            --num-mapper)
                NUM_MAPPER="$2"
                shift
                ;;
            --) break ;;
        esac
        shift
    done

    # input file 검사
    if [ -z "${INPUT_FILE}" ]; then
        print_usage "no input file"
    fi

    # corpus dir 검사
    if [ -z "${CORPUS_DIR}" ]; then
        print_usage "no corpus dir"
    fi

    if [ -z "${RSC_SRC}" ]; then
        RSC_SRC=../../rsc/src
    fi
    if [ -z "${LIB_PATH}" ]; then
        LIB_PATH=../../build/lib/libkhaiii.so
    fi
    if [ -z "${RSC_DIR}" ]; then
        RSC_DIR=../../build/share/khaiii
    fi
    if [ -z "${NUM_MAPPER}" ]; then
        NUM_MAPPER=1000
    fi

    INPUT_FILE=$(abspath ${INPUT_FILE})
    LIB_PATH=$(abspath ${LIB_PATH})
    RSC_DIR=$(abspath ${RSC_DIR})
    RSC_SRC=$(abspath ${RSC_SRC})
    CORPUS_DIR=$(abspath ${CORPUS_DIR})
}


function init_envs() {
    # global variables
    INPUT_DIR=errpatch.in
    OUTPUT_DIR=errpatch.out
    CACHE_DIR=errpatch.cache
}


function split_input() {
    >&2 echo "{{{{{{{{{{ ${FUNCNAME[0]} {{{{{{{{{{"

    local total_line
    total_line=$(wc -l < ${INPUT_FILE})
    local line_per_split=$((total_line / NUM_MAPPER))
    rm -rf ${INPUT_DIR}
    mkdir -p ${INPUT_DIR}
    shuf ${INPUT_FILE} | split -d -a 5 -l ${line_per_split} - ${INPUT_DIR}/part-

    hadoop fs -test -e ${INPUT_DIR} && hadoop fs -rm -skipTrash -r ${INPUT_DIR}
    hadoop fs -put ${INPUT_DIR}

    >&2 echo "}}}}}}}}}} ${FUNCNAME[0]} }}}}}}}}}}"
}


function cache_files() {
    >&2 echo "{{{{{{{{{{ ${FUNCNAME[0]} {{{{{{{{{{"

    hadoop fs -test -e ${CACHE_DIR} && hadoop fs -rm -skipTrash -r ${CACHE_DIR}
    hadoop fs -mkdir -p ${CACHE_DIR}

    hadoop fs -put ../../src/main/python/khaiii ${CACHE_DIR}
    hadoop fs -mkdir -p ${CACHE_DIR}/khaiii/lib
    hadoop fs -put ${LIB_PATH} ${CACHE_DIR}/khaiii/lib

    hadoop fs -mkdir -p ${CACHE_DIR}/khaiii/share
    hadoop fs -put ${RSC_DIR} ${CACHE_DIR}/khaiii/share/khaiii

    hadoop fs -put ${RSC_SRC} ${CACHE_DIR}/rsc_src

    hadoop fs -mkdir -p ${CACHE_DIR}/corpus
    hadoop fs -put ${CORPUS_DIR}/*.txt ${CACHE_DIR}/corpus

    >&2 echo "}}}}}}}}}} ${FUNCNAME[0]} }}}}}}}}}}"
}


function run_hadoop() {
    >&2 echo "{{{{{{{{{{ ${FUNCNAME[0]} {{{{{{{{{{"

    hadoop fs -test -e ${OUTPUT_DIR} && hadoop fs -rm -skipTrash -r ${OUTPUT_DIR}
    yarn jar ${HADOOP_HOME}/share/hadoop/tools/lib/hadoop-streaming-*.jar \
        -D mapred.job.name=validate_errpatch \
        -D mapred.reduce.tasks=0 \
        -cmdenv PYTHONPATH="./${CACHE_DIR}" \
        -file ./validate_errpatch.py \
        -input "${INPUT_DIR}" \
        -output "${OUTPUT_DIR}" \
        -cacheFile "${CACHE_DIR}#${CACHE_DIR}" \
        -mapper "./validate_errpatch.py -c ./${CACHE_DIR}/corpus --rsc-src ./${CACHE_DIR}/rsc_src"

    hadoop fs -text ${OUTPUT_DIR}/part-* > "$(dirname ${INPUT_FILE})/errpatch.valid"

    >&2 echo "}}}}}}}}}} ${FUNCNAME[0]} }}}}}}}}}}"
}


function del_temp() {
    >&2 echo "{{{{{{{{{{ ${FUNCNAME[0]} {{{{{{{{{{"

    hadoop fs -rm -skipTrash -r ${INPUT_DIR} ${OUTPUT_DIR} ${CACHE_DIR}
    rm -rf ${INPUT_DIR}

    >&2 echo "}}}}}}}}}} ${FUNCNAME[0]} }}}}}}}}}}"
}


########
# main #
########
parse_args $@
cd "$(dirname $0)"
init_envs

# split_input
# cache_files
run_hadoop
del_temp
