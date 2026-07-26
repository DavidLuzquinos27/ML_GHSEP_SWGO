#!/bin/bash

INPUT_DIR="/home/acolan/swgo_2024/DATOS/M7_reco_D8_proton_V2"
OUTPUT_DIR="/home/acolan/swgo_2024/DATOS/M7_reco_D8_proton_V2_new_variables"
MUON_TAGGER="/home/acolan/swgo_2024/DATOS/muon_tagger_D8"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
MAX_JOBS=16
READ_JOBS=4


running=0

process_file() {
    local i=$1
    local j=$2

    local num
    num=$(printf "%06d" "$i")

    local infile="${INPUT_DIR}/reco-DAT${num}_D8_proton_${j}_50000.root"
    local outfile="${OUTPUT_DIR}/reco-DAT${num}_D8_proton_${j}_50000.pkl"
    local logfile="${OUTPUT_DIR}/reco-DAT${num}_D8_proton_${j}_50000.log"

    if [ ! -f "$infile" ]; then
        return 0
    fi

    if [ -f "$outfile" ]; then
        echo "Saltando ${outfile} (ya existe)"
        return 0
    fi

    echo "Procesando DAT${num} gamma ${j}"

    python -m pyswgo.muon_tagger.scripts.predict \
        --muon-tagger-path "$MUON_TAGGER" \
        -o "$outfile" \
        --read-jobs "$READ_JOBS" \
        "$infile" 
}

for i in $(seq 1 10000); do
    for j in 0 1 2 3 4; do
        process_file "$i" "$j" &
        ((running++))

        if [ "$running" -ge "$MAX_JOBS" ]; then
            wait -n
            ((running--))
        fi
    done
done

wait
echo "Proceso terminado."
