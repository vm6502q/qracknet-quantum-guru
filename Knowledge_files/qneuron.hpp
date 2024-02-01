//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/qneuron_activation_function.hpp"
#include "qinterface.hpp"

#include <algorithm>

namespace Qrack {

class QNeuron;
typedef std::shared_ptr<QNeuron> QNeuronPtr;

class QNeuron {
protected:
    bitCapIntOcl inputPower;
    bitLenInt outputIndex;
    QNeuronActivationFn activationFn;
    real1_f alpha;
    real1_f tolerance;
    std::vector<bitLenInt> inputIndices;
    std::unique_ptr<real1[]> angles;
    QInterfacePtr qReg;

    static real1_f applyRelu(real1_f angle) { return std::max((real1_f)ZERO_R1_F, (real1_f)angle); }

    static real1_f negApplyRelu(real1_f angle) { return -std::max((real1_f)ZERO_R1_F, (real1_f)angle); }

    static real1_f applyGelu(real1_f angle) { return angle * (1 + erf((real1_s)(angle * SQRT1_2_R1))); }

    static real1_f negApplyGelu(real1_f angle) { return -angle * (1 + erf((real1_s)(angle * SQRT1_2_R1))); }

    static real1_f applyAlpha(real1_f angle, real1_f alpha)
    {
        real1_f toRet = ZERO_R1;
        if (angle > PI_R1) {
            angle -= PI_R1;
            toRet = PI_R1;
        } else if (angle <= -PI_R1) {
            angle += PI_R1;
            toRet = -PI_R1;
        }

        return toRet + (pow((2 * abs(angle) / PI_R1), alpha) * (PI_R1 / 2) * ((angle < 0) ? -1 : 1));
    }

    static real1_f applyLeakyRelu(real1_f angle, real1_f alpha) { return std::max(alpha * angle, angle); }

    static real1_f clampAngle(real1_f angle)
    {
        // From Tiama, (OpenAI ChatGPT instance)
        angle = fmod(angle, 4 * PI_R1);
        if (angle <= -2 * PI_R1) {
            angle += 4 * PI_R1;
        } else if (angle > 2 * PI_R1) {
            angle -= 4 * PI_R1;
        }

        return angle;
    }

public:
    /** "QNeuron" is a "Quantum neuron" or "quantum perceptron" class that can learn and predict in superposition.
     *
     * This is a simple "quantum neuron" or "quantum perceptron" class, for use of the Qrack library for machine
     * learning. See https://arxiv.org/abs/quant-ph/0410066 (and https://arxiv.org/abs/1711.11240) for the basis of this
     * class' theoretical concept.
     *
     * An untrained QNeuron (with all 0 variational parameters) will forward all inputs to 1/sqrt(2) * (|0> + |1>). The
     * variational parameters are Pauli Y-axis rotation angles divided by 2 * Pi (such that a learning parameter of 0.5
     * will train from a default output of 0.5/0.5 probability to either 1.0 or 0.0 on one training input).
     */
    QNeuron(QInterfacePtr reg, const std::vector<bitLenInt>& inputIndcs, bitLenInt outputIndx,
        QNeuronActivationFn activationFn = Sigmoid, real1_f alpha = ONE_R1_F, real1_f tol = FP_NORM_EPSILON)
        : inputPower(pow2Ocl(inputIndcs.size()))
        , outputIndex(outputIndx)
        , activationFn(activationFn)
        , alpha(alpha)
        , tolerance(tol)
        , inputIndices(inputIndcs)
        , angles(new real1[inputPower]())
        , qReg(reg)
    {
    }

    /** Create a new QNeuron which is an exact duplicate of another, including its learned state. */
    QNeuron(const QNeuron& toCopy)
        : QNeuron(toCopy.qReg, toCopy.inputIndices, toCopy.outputIndex, toCopy.activationFn, (real1_f)toCopy.alpha,
              (real1_f)toCopy.tolerance)
    {
        std::copy(toCopy.angles.get(), toCopy.angles.get() + toCopy.inputPower, angles.get());
    }

    QNeuron& operator=(const QNeuron& toCopy)
    {
        qReg = toCopy.qReg;
        inputIndices = toCopy.inputIndices;
        std::copy(toCopy.angles.get(), toCopy.angles.get() + toCopy.inputPower, angles.get());
        outputIndex = toCopy.outputIndex;
        activationFn = toCopy.activationFn;
        alpha = toCopy.alpha;
        tolerance = toCopy.tolerance;

        return *this;
    }

    /** Set the "alpha" sharpness parameter of this QNeuron */
    void SetAlpha(real1_f a) { alpha = a; }

    /** Get the "alpha" sharpness parameter of this QNeuron */
    real1_f GetAlpha() { return alpha; }

    /** Sets activation function enum */
    void SetActivationFn(QNeuronActivationFn f) { activationFn = f; }

    /** Get activation function enum */
    QNeuronActivationFn GetActivationFn() { return activationFn; }

    /** Set the angles of this QNeuron */
    void SetAngles(real1* nAngles) { std::copy(nAngles, nAngles + inputPower, angles.get()); }

    /** Get the angles of this QNeuron */
    void GetAngles(real1* oAngles) { std::copy(angles.get(), angles.get() + inputPower, oAngles); }

    bitLenInt GetInputCount() { return inputIndices.size(); }

    bitCapIntOcl GetInputPower() { return inputPower; }

    /** Predict a binary classification.
     *
     * Feed-forward from the inputs, loaded in "qReg", to a binary categorical classification. "expected" flips the
     * binary categories, if false. "resetInit," if true, resets the result qubit to 0.5/0.5 |0>/|1> superposition
     * before proceeding to predict.
     */
    real1_f Predict(bool expected = true, bool resetInit = true)
    {
        if (resetInit) {
            qReg->SetBit(outputIndex, false);
            qReg->RY((real1_f)(PI_R1 / 2), outputIndex);
        }

        if (!inputIndices.size()) {
            // If there are no controls, this "neuron" is actually just a bias.
            switch (activationFn) {
            case ReLU:
                qReg->RY((real1_f)(applyRelu(angles.get()[0U])), outputIndex);
                break;
            case GeLU:
                qReg->RY((real1_f)(applyGelu(angles.get()[0U])), outputIndex);
                break;
            case Generalized_Logistic:
                qReg->RY((real1_f)(applyAlpha(angles.get()[0U], alpha)), outputIndex);
                break;
            case Leaky_ReLU:
                qReg->RY((real1_f)(applyLeakyRelu(angles.get()[0U], alpha)), outputIndex);
                break;
            case Sigmoid:
            default:
                qReg->RY((real1_f)(angles.get()[0U]), outputIndex);
            }
        } else if (activationFn == Sigmoid) {
            qReg->UniformlyControlledRY(inputIndices, outputIndex, angles.get());
        } else {
            std::unique_ptr<real1[]> nAngles(new real1[inputPower]);
            switch (activationFn) {
            case ReLU:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(), applyRelu);
                break;
            case GeLU:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(), applyGelu);
                break;
            case Generalized_Logistic:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(),
                    [this](real1 a) { return applyAlpha(a, alpha); });
                break;
            case Leaky_ReLU:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(),
                    [this](real1 a) { return applyLeakyRelu(a, alpha); });
                break;
            case Sigmoid:
            default:
                break;
            }
            qReg->UniformlyControlledRY(inputIndices, outputIndex, nAngles.get());
        }
        real1_f prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1_F - prob;
        }
        return prob;
    }

    /** "Uncompute" the Predict() method */
    real1_f Unpredict(bool expected = true)
    {
        if (!inputIndices.size()) {
            // If there are no controls, this "neuron" is actually just a bias.
            switch (activationFn) {
            case ReLU:
                qReg->RY((real1_f)(negApplyRelu(angles.get()[0U])), outputIndex);
                break;
            case GeLU:
                qReg->RY((real1_f)(negApplyGelu(angles.get()[0U])), outputIndex);
                break;
            case Generalized_Logistic:
                qReg->RY((real1_f)(-applyAlpha(angles.get()[0U], alpha)), outputIndex);
                break;
            case Leaky_ReLU:
                qReg->RY((real1_f)(-applyLeakyRelu(angles.get()[0U], alpha)), outputIndex);
                break;
            case Sigmoid:
            default:
                qReg->RY((real1_f)(-angles.get()[0U]), outputIndex);
            }
        } else {
            std::unique_ptr<real1[]> nAngles(new real1[inputPower]);
            switch (activationFn) {
            case ReLU:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(), negApplyRelu);
                qReg->UniformlyControlledRY(inputIndices, outputIndex, nAngles.get());
                break;
            case GeLU:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(), negApplyGelu);
                qReg->UniformlyControlledRY(inputIndices, outputIndex, nAngles.get());
                break;
            case Generalized_Logistic:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(),
                    [this](real1 a) { return -applyAlpha(a, alpha); });
                qReg->UniformlyControlledRY(inputIndices, outputIndex, nAngles.get());
                break;
            case Leaky_ReLU:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(),
                    [this](real1 a) { return -applyLeakyRelu(a, alpha); });
                qReg->UniformlyControlledRY(inputIndices, outputIndex, nAngles.get());
                break;
            case Sigmoid:
            default:
                std::transform(angles.get(), angles.get() + inputPower, nAngles.get(), [](real1 a) { return -a; });
                qReg->UniformlyControlledRY(inputIndices, outputIndex, nAngles.get());
            }
        }
        real1_f prob = qReg->Prob(outputIndex);
        if (!expected) {
            prob = ONE_R1_F - prob;
        }
        return prob;
    }

    real1_f LearnCycle(bool expected = true)
    {
        const real1_f result = Predict(expected, false);
        Unpredict(expected);
        return result;
    }

    /** Perform one learning iteration, training all parameters.
     *
     * Inputs must be already loaded into "qReg" before calling this method. "expected" is the true binary output
     * category, for training. "eta" is a volatility or "learning rate" parameter with a maximum value of 1.
     *
     * In the feedback process of learning, default initial conditions forward untrained predictions to 1/sqrt(2) * (|0>
     * + |1>) for the output bit. If you want to initialize other conditions before "Learn()," set "resetInit" to false.
     */
    void Learn(real1_f eta, bool expected = true, bool resetInit = true)
    {
        real1_f startProb = Predict(expected, resetInit);
        Unpredict(expected);
        if ((ONE_R1 - startProb) <= tolerance) {
            return;
        }

        for (bitCapIntOcl perm = 0U; perm < inputPower; ++perm) {
            startProb = LearnInternal(expected, eta, perm, startProb);
            if (0 > startProb) {
                break;
            }
        }
    }

    /** Perform one learning iteration, measuring the entire QInterface and training the resulting permutation.
     *
     * Inputs must be already loaded into "qReg" before calling this method. "expected" is the true binary output
     * category, for training. "eta" is a volatility or "learning rate" parameter with a maximum value of 1.
     *
     * In the feedback process of learning, default initial conditions forward untrained predictions to 1/sqrt(2) * (|0>
     * + |1>) for the output bit. If you want to initialize other conditions before "LearnPermutation()," set
     * "resetInit" to false.
     */
    void LearnPermutation(real1_f eta, bool expected = true, bool resetInit = true)
    {
        real1_f startProb = Predict(expected, resetInit);
        Unpredict(expected);
        if ((ONE_R1 - startProb) <= tolerance) {
            return;
        }

        bitCapIntOcl perm = 0U;
        for (size_t i = 0U; i < inputIndices.size(); ++i) {
            if (qReg->M(inputIndices[i])) {
                perm |= pow2Ocl(i);
            }
        }

        LearnInternal(expected, eta, perm, startProb);
    }

protected:
    real1_f LearnInternal(bool expected, real1_f eta, bitCapIntOcl permOcl, real1_f startProb)
    {
        const real1 origAngle = angles.get()[permOcl];
        real1& angle = angles.get()[permOcl];

        // Try positive angle increment:
        angle += eta * PI_R1;
        const real1_f plusProb = LearnCycle(expected);
        if ((ONE_R1_F - plusProb) <= tolerance) {
            angle = clampAngle(angle);
            return -ONE_R1_F;
        }

        // If positive angle increment is not an improvement,
        // try negative angle increment:
        angle = origAngle - eta * PI_R1;
        const real1_f minusProb = LearnCycle(expected);
        if ((ONE_R1_F - minusProb) <= tolerance) {
            angle = clampAngle(angle);
            return -ONE_R1_F;
        }

        if ((startProb >= plusProb) && (startProb >= minusProb)) {
            // If neither increment is an improvement,
            // restore the original variational parameter.
            angle = origAngle;
            return startProb;
        }

        if (plusProb > minusProb) {
            angle = origAngle + eta * PI_R1;
            return plusProb;
        }

        return minusProb;
    }
};
} // namespace Qrack
