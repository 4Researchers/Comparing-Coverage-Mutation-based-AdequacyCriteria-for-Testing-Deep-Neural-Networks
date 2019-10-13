#!/usr/bin/zsh

set mutatedModelsPath ../build-in-resource/mutated_models/mnist/lenet/gf/5e-2p/
set test_result_folder ../lcr_auc-testing-results/mnist/lenet/gf/5e-2p/fgsm/
set nrLcrPath ../build-in-resource/nr-lcr/mnsit/lenet/gf/5e-2p/nrLCR.npy
for j in {0..9}; do
    for i in {0..100}; do
    # spawn /home/ict520c/Documents/icse2019/scripts/newname.sh
    # expect {
    #     "choice" { send -- "n\n" }
    #     "datatype" { send -- "0\n" }
    #     "device" { send -- "-1\n" }
    #     "useTrainData" { send -- "False\n" }
    #     "batchModelSize" { send -- "2\n" }
    #     "mutatedModelsPath" { send -- "$mutatedModelPath\n" }
    #     "testSamplesPath" { send -- "../build-in-resource/dataset/mnist/adversarial/fgsm/a$i\n" }
    #     "seedModelName" { send -- "lenet\n" }
    #     "test_result_folder" { send -- "$test_result_folder\n" }
    #     "maxModelsUsed" { send -- "10\n" }
    #     "choice" { send -- "y\n" }
    #     "nrLcrPath" { send -- "$nrLcrPath\n" }
    #     "char\n" { send -- "\n" }
    # }
    # expect eof
        echo n  > input.txt
        echo 0 >> input.txt
        echo 0 >> input.txt
        echo adv >> input.txt
        echo False >> input.txt
        echo 2 >> input.txt
        echo ../build-in-resource/mutated_models/mnist/lenet/ >> input.txt
        echo ../build-in-resource/dataset/mnist/adversarial/jsma/r$j/a$i >> input.txt
        echo lenet >> input.txt
        echo ../lcr_auc-testing-results/mnist/lenet/gf/5e-2p/jsma/ >> input.txt
        echo 10 >> input.txt
        echo y >> input.txt
        echo ../build-in-resource/nr-lcr/mnsit/lenet/gf/5e-2p/nrLCR.npy >> input.txt
        ./lcr_acu_analysis.sh < input.txt
    done
done
