import {
    loadTxt,
    vecMake,
    matMake,
    vecShow,
    matShow,
    argmax,
    arange,
    Erratic,
    hyperTan,
    logSig,
    vecMax,
    softmax
} from "../mathlib/utilities"

class NeuralNet {
    constructor(numInput, numHidden, numOutput,seed) {
        this.rnd = new Erratic(seed)
        this.ni = numInput
        this.nh = numHidden
        this.no = numOutput
        
        this.iNodes = vecMake(this.ni, 0.0)
        this.hNodes = vecMake(this.nh, 0.0)
        this.oNodes = vecMake(this.no, 0.0)

        this.ihWeights = matMake(this.ni, this.nh, 0.0)
        this.hoWeights = matMake(this.nh, this.no, 0.0)

        this.hBiases = vecMake(this.nh, 0.0)
        this.oBiases = vecMake(this.no, 0.0)
        this.initWeights()
    }

    initWeights() {
        let lo = -0.01
        let hi = 0.01
        for(let i = 0; i < this.ni;++i) {
            for (let j = 0; j < this.nh; ++j) {
                this.ihWeights[i][j] = (hi - lo)*this.rnd.next() + lo
            }
        }

        for(let j = 0;j < this.nh;++j) {
            for(let k =0;k < this.no;++k) {
                this.hoWeights[j][k] = (hi-lo)*this.rnd.next() + lo
            }
        }
    }

    eval(x) {
        let hSums = vecMake(this.nh, 0.0)
        let oSums = vecMake(this.no, 0.0)
        this.iNodes = x

        for (let j = 0; j < this.nh; ++j) {
            for (let i = 0; i < this.ni; ++i) {
                /// 计算隐藏层中j节点的值
                hSums[j] += this.iNodes[i] * this.ihWeights[i][j]
            }
            // 使用偏置补偿(k = ax + b)
            hSums[j] += this.hBiases[j]
            this.hNodes[j] = hyperTan(hSums[j])
        }
        // console.log("\n Internal hidden node values = ")
        // vecShow(this.hNodes, 4)

        for (let k = 0; k < this.no; ++k) {
            for (let j = 0; j < this.nh; ++j) {
                oSums[k] += this.hNodes[j] * this.hoWeights[j][k]
            }
            oSums[k] += this.oBiases[k]
        }

        this.oNodes = softmax(oSums)
        // console.log("\nInternal softmax output nodes = ")
        // vecShow(this.oNodes, 4)

        let result = []
        for (let k = 0; k < this.no; ++k) {
            result[k] = this.oNodes[k]
        }

        return result
    } // eval()
    /**
     * @param {Array<number>} wts
     * wts --Order--> [ihWts,hBiases,hoWts,oBiases]
     */
    setWeights(wts) {
        let p = 0
        for (let i = 0; i < this.ni; ++i) {
            for (let j = 0; j < this.nh; ++j) {
                this.ihWeights[i][j] = wts[p++]
            }
        }

        for (let j = 0; j < this.nh; ++j) {
            this.hBiases[j] = wts[p++]
        }

        for (let j = 0; j<this.nh; ++j) {
            for (let k = 0; k < this.no; ++k) {
                this.hoWeights[j][k] = wts[p++]
            }
        }

        for (let k = 0; k < this.no; ++k) {
            this.oBiases[k] = wts[p++]
        }
    }

    getWeights() {
        let numWts = (this.ni * this.nh) + this.nh + (this.nh * this.no) + this.no
        let result = vecMake(numWts,0.0)
        let p = 0
        for(let i = 0; i < this.ni;++i) {
            for(let j = 0;j < this.nh;++j) {
                result[p++]  =this.ihWeights[i][j]
            }
        }

        for(let j = 0;j < this.nh;++j) {
            result[p++] = this.hBiases[j]
        }

        for(let j = 0;j < this.nh;++j) {
            for(let k = 0;k < this.no;++k){
                result[p++] = this.hoWeights[j][k]
            }
        }

        for(let k = 0;k < this.no;++k) {
            result[p++] = this.oBiases[k]
        }

        return result
    }
    /**
     * 
     * @param {Array<any>} v 
     */
    shuffer(v) {
        let n = v.length
        for(let i = 0;i < n; ++i) {
            let r = this.rnd.nextInt(i,n)
            let tmp = v[r]
            v[r] = v[i]
            v[i] = tmp
        }
    }

    train(trainX,trainY,lrnRate,maxEpochs) {
        let hoGrads = matMake(this.nh,this.no,0.0)
        let obGrads = vecMake(this.no,0.0)
        let ihGrads = matMake(this.ni,this.nh,0.0)
        let hbGrads = vecMake(this.nh,0.0)

        let oSignals = vecMake(this.no,0.0)
        let hSignals = vecMake(this.nh,0.0)

        let n = trainX.length
        let indices = arange(n)
        let freq = Math.trunc(maxEpochs / 10)

        for(let epoch = 0; epoch < maxEpochs; ++epoch) {
            this.shuffer(indices)
            for(let ii =0;ii < n;++ii) {
                let idx = indices[ii]
                let X = trainX[idx]
                let Y = trainY[idx]
                this.eval(X)
                /// compute output signals
                for(let k = 0;k < this.no;++k) {
                    /// 使用 Softmax
                    let derivative = (1-this.oNodes[k]) * this.oNodes[k]
                    /// E = (t-o)^2 do E' = (o-t)
                    oSignals[k] = derivative * (this.oNodes[k] - Y[k])
                }
                /// compute hidden-to-output weight gradients using output signals
                for(let j = 0; j < this.nh;++j) {
                    for(let k =0;k < this.no;++k) {
                        hoGrads[j][k] = oSignals[k] * this.hNodes[j]
                    }
                }

                /// cmopute output node bias gradients using output signals
                for(let k = 0; k < this.no;++k) {
                    obGrads[k] = oSignals[k] * 1.0
                }

                /// compute hidden node signals
                for(let j = 0;j < this.nh;++j) {
                    let sum = 0.0
                    for(let k = 0;k < this.no;++k) {
                        sum += oSignals[k] * this.hoWeights[j][k]
                    }
                    let derivative = (1-this.hNodes[j]) * (1 + this.hNodes[j])// tanh
                    hSignals[j] = derivative * sum
                }

                // cmpute input-to-hidden weight gradients using hidden signals
                for(let i = 0;i<this.ni;++i) {
                    for(let j = 0;j < this.nh;++j) {
                        ihGrads[i][j] = hSignals[j] * this.iNodes[i]
                    }
                }
                // compute hidden node bias gradients using hidden signals
                for( let j = 0; j < this.nh;++j) [
                    hbGrads[j] = hSignals[j] * 1.0
                ]
                // update input-to-hidden weights
                for(let i = 0;i < this.ni;++i) {
                    for(let j = 0;j < this.nh;++j) {
                        let delta = -1.0 * lrnRate * ihGrads[i][j]
                        this.ihWeights[i][j] += delta
                    }
                }
                // update hidden node biases
                for(let j = 0;j < this.nh;++j) {
                    let delta = -1.0 * lrnRate * hbGrads[j]
                    this.hBiases[j] += delta
                }

                /// update hidden-to-output weights
                for(let j =0;j < this.nh;++j) {
                    for(let k = 0;k < this.no;++k) {
                        let delta = -1.0* lrnRate * hoGrads[j][k]
                        this.hoWeights[j][k] += delta
                    }
                }

                // update output node biases
                for(let k = 0;k < this.no;++k) {
                    let delta = -1.0 * lrnRate * obGrads[k]
                    this.oBiases[k] += delta
                }
            } // ii

            if(epoch % freq === 0) {
                let mse = this.meanSqErr(trainX,trainY).toFixed(4)
                let acc = this.accuracy(trainX,trainY).toFixed(4)

                let s1 = "epoch: " + epoch.toString()
                let s2 = "MSE = " + mse.toString()
                let s3 = " acc = " + acc.toString()

                console.log(s1 + s2 + s3)
            }
        } // epoch
    }// train

    meanCrossEntErr(dataX,dataY) {
        let sumCEE = 0.0;
        for(let i =0;i < dataX.length; ++i) {
            let X = dataX[i]
            let Y = dataY[i]
            let oupt = this.eval(X)
            let idx = argmax(Y)
            sumCEE += Math.log(oupt[idx])
        }
        sumCEE *= -1
        return sumCEE / dataX.length
    }// meanCrossEntErr

    meanSqErr(dataX,dataY){
        let sumSE = 0.0
        for(let i = 0;i < dataX.length;++i) { // each data item
            let X = dataX[i]
            let Y = dataY[i] // target output lik(0,1,0)
            let oupt = this.eval(X) // cmputed like(0.23,0.66,0.11)
            for(let k = 0;k < this.no;++k) {
                let err = Y[k] - oupt[k]// target - computed
                sumSE += err ** 2
            }
        }
        return sumSE / dataX.length
    }// meanSqErr()


    accuracy(dataX,dataY) {
        let nc = 0
        let nw = 0
        for(let i = 0;i < dataX.length; ++i) {
            let X = dataX[i]
            let Y = dataY[i]
            let oupt = this.eval(X)
            let computedIdx = argmax(oupt)
            let targetIdx = argmax(Y)
            if(computedIdx === targetIdx) {
                ++nc
            } else{
                ++nw
            }
        }
        return nc / (nc + nw)
    }// accuracy()


}

function main() {
    process.stdout.write("\\033[0m")
    process.stdout.write("\\x1b[1m" + "\\x1b[37m")
    console.log("\\nBegin IO demo ")

    let trainX = loadTxt(__dirname+'/Data/iris_train.txt',',',[0,1,2,3])
    let trainY = loadTxt(__dirname+'/Data/iris_train.txt',',',[4,5,6])
    let testX = loadTxt(__dirname+'/Data/iris_test.txt',',',[0,1,2,3])
    let testY = loadTxt(__dirname+'/Data/iris_test.txt',',',[4,5,6])
    // 2. create network
    console.log("\\nCreating 3-4-2 neural net")
    let seed = 0
    let nn = new NeuralNet(4, 7, 3,seed)
    // 3. train network
    let lrnRate = 0.01
    let maxEpochs = 50
    console.log("\nStarting training with learning rate = 0.01")
    nn.train(trainX,trainY,lrnRate,maxEpochs)
    console.log("Training complete")

    // 4. evaluate model
    let trainAcc = nn.accuracy(trainX,trainY)
    let testAcc = nn.accuracy(testX,testY)

    console.log("\nAccuracy on training data = " + trainAcc.toFixed(4).toString())
    console.log("Accuracy on test data = " + testAcc.toFixed(4).toString())
    // 5. use trained model
    let unknown = [5.1,3.1,4.1,2.1]
    let predicated = nn.eval(unknown)
    console.log("\nSetting feature of iris to predict: ",vecShow(unknown,1,12))
    console.log("\nPredicated quasi-probabliteies: "+ vecShow(predicated,4,12))

    process.stdout.write("\\033[0m")
    console.log("\nEnd demo")
}

main()