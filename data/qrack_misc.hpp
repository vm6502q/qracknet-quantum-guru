//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2024. All rights reserved.
//
// Qrack is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// This "qrack_misc.hpp" file is actually a curated pseudo-code text compendium of
// major algorithms in Qrack, not a compilable file. It is meant highlight both 
// novel solutions and new pareto-efficient algorithms over prior state-of-the-art.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "common/qneuron_activation_function.hpp"
#include "qinterface.hpp"

#include <algorithm>

namespace Qrack {
/**
 * Enumerated list of Pauli bases
 */
enum Pauli {
    /// Pauli Identity operator. Corresponds to Q# constant "PauliI."
    PauliI = 0,
    /// Pauli X operator. Corresponds to Q# constant "PauliX."
    PauliX = 1,
    /// Pauli Y operator. Corresponds to Q# constant "PauliY."
    PauliY = 3,
    /// Pauli Z operator. Corresponds to Q# constant "PauliZ."
    PauliZ = 2
};

/**
 * Enumerated list of activation functions
 */
enum QNeuronActivationFn {
    /// Default
    Sigmoid = 0,
    /// Rectified linear
    ReLU = 1,
    /// Gaussian linear
    GeLU = 2,
    /// Version of (default) "Sigmoid" with tunable sharpness
    Generalized_Logistic = 3,
    /// Leaky rectified linear
    Leaky_ReLU = 4
};

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
        QNeuronActivationFn activationFn = Sigmoid, real1_f alpha = ONE_R1_F, real1_f tol = FP_NORM_EPSILON / 2)
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

        if (inputIndices.empty()) {
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
        if (inputIndices.empty()) {
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

//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2024. All rights reserved.
//
// File: "qcircuit.hpp"
//
// Qrack's concept for a quantum "circuit" class "locally optimizes" by default and
// breaks _all_ gates down into multiplexers with single target qubits, very similar
// to the `UniformlyControlledRZ()` gates of QNeuron, except with general 2-by-2
// complex unitary payloads instead of single RZ parameters, for every possible
// combination of control qubits.
//
// Observe how practically computationally simple commutation and composition rules
// become! Consider that we're only working with maybe less than 50 to about 200
// logical qubits at once, even in actual NISQ hardware. Single-target-qubit
// multiplexers end up easily (already, for a long time) empirically demonstrating
// perfect reduction of random mirror circuits to identity operator on at least these
// qubit scales, without reliance on any specific knowledge that these are mirror
// circuits.
//
// It might be unnecessary to use the full machinery of LLVM to try to optimize on
// order of up to 200 _bits,_ when Qrack has found it's easier to simply develop
// commutation and composition rules for fully-numerical single-target-qubit
// multiplexer operators (though not with so naive an implementation as the full
// dimension of the general unitary matrix representation of these operators). Then,
// it's really "just floating-point _numerics,_" to simplify a circuit, and the scaling
// does not come _close_ to exhausting our capacities for floating-point numerics, even
// on a laptop.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include "qinterface.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <list>

#define amp_leq_0(x) (norm(x) <= FP_NORM_EPSILON)
#define __IS_REAL_1(r) (abs(ONE_R1 - r) <= FP_NORM_EPSILON)
#define __IS_SAME(a, b) (norm((a) - (b)) <= FP_NORM_EPSILON)
#define __IS_CTRLED_CLIFFORD(top, bottom)                                                                              \
    ((__IS_REAL_1(std::real(top)) || __IS_REAL_1(std::imag(bottom))) &&                                                \
        (__IS_SAME(top, bottom) || __IS_SAME(top, -bottom)))
#define __IS_CLIFFORD_PHASE_INVERT(top, bottom)                                                                        \
    (__IS_SAME(top, bottom) || __IS_SAME(top, -bottom) || __IS_SAME(top, I_CMPLX * bottom) ||                          \
        __IS_SAME(top, -I_CMPLX * bottom))
#define __IS_CLIFFORD(mtrx)                                                                                            \
    ((__IS_PHASE(mtrx) && __IS_CLIFFORD_PHASE_INVERT(mtrx[0], mtrx[3])) ||                                             \
        (__IS_INVERT(mtrx) && __IS_CLIFFORD_PHASE_INVERT(mtrx[1], mtrx[2])) ||                                         \
        ((__IS_SAME(mtrx[0U], mtrx[1U]) || __IS_SAME(mtrx[0U], -mtrx[1U]) ||                                           \
             __IS_SAME(mtrx[0U], I_CMPLX * mtrx[1U]) || __IS_SAME(mtrx[0U], -I_CMPLX * mtrx[1U])) &&                   \
            (__IS_SAME(mtrx[0U], mtrx[2U]) || __IS_SAME(mtrx[0U], -mtrx[2U]) ||                                        \
                __IS_SAME(mtrx[0U], I_CMPLX * mtrx[2U]) || __IS_SAME(mtrx[0U], -I_CMPLX * mtrx[2U])) &&                \
            (__IS_SAME(mtrx[0U], mtrx[3U]) || __IS_SAME(mtrx[0U], -mtrx[3U]) ||                                        \
                __IS_SAME(mtrx[0U], I_CMPLX * mtrx[3U]) || IS_SAME(mtrx[0U], -I_CMPLX * mtrx[3U]))))
#define __IS_PHASE(mtrx) (IS_NORM_0(mtrx[1U]) && IS_NORM_0(mtrx[2U]))
#define __IS_INVERT(mtrx) (IS_NORM_0(mtrx[0U]) && IS_NORM_0(mtrx[3U]))

namespace Qrack {

/**
 * Single gate in `QCircuit` definition
 */
struct QCircuitGate;
typedef std::shared_ptr<QCircuitGate> QCircuitGatePtr;

struct QCircuitGate {
    bitLenInt target;
    std::map<bitCapInt, std::shared_ptr<complex>> payloads;
    std::set<bitLenInt> controls;

    /**
     * Identity gate constructor
     */
    QCircuitGate()
        : target(0)
        , payloads()
        , controls()

    {
        Clear();
    }

    /**
     * `Swap` gate constructor
     */
    QCircuitGate(bitLenInt q1, bitLenInt q2)
        : target(q1)
        , payloads()
        , controls({ q2 })

    {
        // Swap gate constructor.
    }

    /**
     * Single-qubit gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[])
        : target(trgt)
    {
        payloads[ZERO_BCI] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4, payloads[ZERO_BCI].get());
    }

    /**
     * Controlled gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[], const std::set<bitLenInt>& ctrls, const bitCapInt& perm)
        : target(trgt)
        , controls(ctrls)
    {
        const std::shared_ptr<complex>& p = payloads[perm] =
            std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4, p.get());
    }

    /**
     * Uniformly controlled gate constructor (that only accepts control qubits is ascending order)
     */
    QCircuitGate(
        bitLenInt trgt, const std::map<bitCapInt, std::shared_ptr<complex>>& pylds, const std::set<bitLenInt>& ctrls)
        : target(trgt)
        , controls(ctrls)
    {
        for (const auto& payload : pylds) {
            payloads[payload.first] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4, payloads[payload.first].get());
        }
    }

    QCircuitGatePtr Clone() { return std::make_shared<QCircuitGate>(target, payloads, controls); }

    /**
     * Can I combine myself with gate `other`?
     */
    bool CanCombine(QCircuitGatePtr other, bool clifford = false)
    {
        if (target != other->target) {
            return false;
        }

        if (controls.empty() && other->controls.empty()) {
            return true;
        }

        if (clifford) {
            const bool mc = IsClifford();
            const bool oc = other->IsClifford();

            if (mc != oc) {
                return false;
            }

            if (mc) {
                return controls.empty() || other->controls.empty() ||
                    (*(controls.begin()) == *(other->controls.begin()));
            }
        }

        if (std::includes(other->controls.begin(), other->controls.end(), controls.begin(), controls.end()) ||
            std::includes(controls.begin(), controls.end(), other->controls.begin(), other->controls.end())) {
            return true;
        }

        return false;
    }

    /**
     * Set this gate to the identity operator.
     */
    void Clear()
    {
        controls.clear();
        payloads.clear();

        payloads[ZERO_BCI] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        complex* p = payloads[ZERO_BCI].get();
        p[0] = ONE_CMPLX;
        p[1] = ZERO_CMPLX;
        p[2] = ZERO_CMPLX;
        p[3] = ONE_CMPLX;
    }

    /**
     * Add control qubit.
     */
    void AddControl(bitLenInt c)
    {
        if (controls.find(c) != controls.end()) {
            return;
        }

        controls.insert(c);

        const size_t cpos = std::distance(controls.begin(), controls.find(c));
        const bitCapInt midPow = pow2(cpos);
        bitCapInt lowMask = midPow;
        bi_decrement(&lowMask, 1U);
        const bitCapInt highMask = ~lowMask;

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            bitCapInt nKey = (payload.first & lowMask) | ((payload.first & highMask) << 1U);

            nPayloads.emplace(nKey, payload.second);

            std::shared_ptr<complex> np = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4U, np.get());
            bi_or_ip(&nKey, midPow);
            nPayloads.emplace(nKey, np);
        }

        payloads = nPayloads;
    }

    /**
     * Check if a control qubit can be removed.
     */
    bool CanRemoveControl(bitLenInt c)
    {
        const size_t cpos = std::distance(controls.begin(), controls.find(c));
        const bitCapInt midPow = pow2(cpos);

        for (const auto& payload : payloads) {
            bitCapInt nKey = ~midPow & payload.first;

            if (bi_compare(nKey, payload.first) == 0) {
                if (payloads.find(nKey | midPow) == payloads.end()) {
                    return false;
                }
            } else {
                if (payloads.find(nKey) == payloads.end()) {
                    return false;
                }
            }

            const complex* l = payloads[nKey].get();
            bi_or_ip(&nKey, midPow);
            const complex* h = payloads[nKey].get();
            if (amp_leq_0(l[0] - h[0]) && amp_leq_0(l[1] - h[1]) && amp_leq_0(l[2] - h[2]) && amp_leq_0(l[3] - h[3])) {
                continue;
            }

            return false;
        }

        return true;
    }

    /**
     * Remove control qubit.
     */
    void RemoveControl(bitLenInt c)
    {
        const size_t cpos = std::distance(controls.begin(), controls.find(c));
        const bitCapInt midPow = pow2(cpos);
        bitCapInt lowMask = midPow;
        bi_decrement(&lowMask, 1U);
        const bitCapInt highMask = ~(lowMask | midPow);

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            nPayloads.emplace((payload.first & lowMask) | ((payload.first & highMask) >> 1U), payload.second);
        }

        payloads = nPayloads;
        controls.erase(c);
    }

    /**
     * Check if I can remove control, and do so, if possible
     */
    bool TryRemoveControl(bitLenInt c)
    {
        if (!CanRemoveControl(c)) {
            return false;
        }
        RemoveControl(c);

        return true;
    }

    /**
     * Combine myself with gate `other`
     */
    void Combine(QCircuitGatePtr other)
    {
        std::set<bitLenInt> ctrlsToTest;
        std::set_intersection(controls.begin(), controls.end(), other->controls.begin(), other->controls.end(),
            std::inserter(ctrlsToTest, ctrlsToTest.begin()));

        if (controls.size() < other->controls.size()) {
            for (const bitLenInt& oc : other->controls) {
                AddControl(oc);
            }
        } else if (controls.size() > other->controls.size()) {
            for (const bitLenInt& c : controls) {
                other->AddControl(c);
            }
        }

        for (const auto& payload : other->payloads) {
            const auto& pit = payloads.find(payload.first);
            if (pit == payloads.end()) {
                const std::shared_ptr<complex>& p = payloads[payload.first] =
                    std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
                std::copy(payload.second.get(), payload.second.get() + 4U, p.get());

                continue;
            }

            complex* p = pit->second.get();
            complex out[4];
            mul2x2(payload.second.get(), p, out);

            if (amp_leq_0(out[1]) && amp_leq_0(out[2]) && amp_leq_0(ONE_CMPLX - out[0]) &&
                amp_leq_0(ONE_CMPLX - out[3])) {
                payloads.erase(pit);

                continue;
            }

            std::copy(out, out + 4U, p);
        }

        if (payloads.empty()) {
            Clear();
            return;
        }

        for (const bitLenInt& c : ctrlsToTest) {
            TryRemoveControl(c);
        }
    }

    /**
     * Check if I can combine with gate `other`, and do so, if possible
     */
    bool TryCombine(QCircuitGatePtr other, bool clifford = false)
    {
        if (!CanCombine(other, clifford)) {
            return false;
        }
        Combine(other);

        return true;
    }

    /**
     * Am I an identity gate?
     */
    bool IsIdentity()
    {
        if (controls.size()) {
            return false;
        }

        if (payloads.size() != 1U) {
            return false;
        }

        complex* p = payloads.begin()->second.get();

        if (amp_leq_0(p[1]) && amp_leq_0(p[2]) && amp_leq_0(ONE_CMPLX - p[0]) && amp_leq_0(ONE_CMPLX - p[3])) {
            return true;
        }

        return false;
    }

    /**
     * Am I a phase gate?
     */
    bool IsPhase()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1]) > FP_NORM_EPSILON) || (norm(p[2]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a Pauli X plus a phase gate?
     */
    bool IsInvert()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a combination of "phase" and "invert" payloads?
     */
    bool IsPhaseInvert()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if (((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON)) &&
                ((norm(p[1]) > FP_NORM_EPSILON) || (norm(p[2]) > FP_NORM_EPSILON))) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a CNOT gate?
     */
    bool IsCnot()
    {
        if ((controls.size() != 1U) || (payloads.size() != 1U) || (payloads.find(ONE_BCI) == payloads.end())) {
            return false;
        }
        complex* p = payloads[ONE_BCI].get();
        if ((norm(p[0]) > FP_NORM_EPSILON) || (norm(p[3]) > FP_NORM_EPSILON) ||
            (norm(ONE_CMPLX - p[1]) > FP_NORM_EPSILON) || (norm(ONE_CMPLX - p[2]) > FP_NORM_EPSILON)) {
            return false;
        }

        return true;
    }

    /**
     * Am I a Clifford gate?
     */
    bool IsClifford()
    {
        if (payloads.empty()) {
            // Swap gate is Clifford
            return true;
        }

        if (controls.size() > 1U) {
            return false;
        }

        if (controls.empty()) {
            return __IS_CLIFFORD(payloads[ZERO_BCI].get());
        }

        for (const auto& kvPair : payloads) {
            const complex* p = kvPair.second.get();
            if ((norm(p[1U]) <= FP_NORM_EPSILON) && (norm(p[2U]) <= FP_NORM_EPSILON)) {
                // Phase payload
                if (!__IS_CLIFFORD_PHASE_INVERT(p[0U], p[3U])) {
                    return false;
                }
            } else if ((norm(p[0U]) <= FP_NORM_EPSILON) && (norm(p[3U]) <= FP_NORM_EPSILON)) {
                // Negation payload
                if (!__IS_CLIFFORD_PHASE_INVERT(p[1U], p[2U])) {
                    return false;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    /**
     * Do I commute with gate `other`?
     */
    bool CanPass(QCircuitGatePtr other)
    {
        const std::set<bitLenInt>::iterator c = other->controls.find(target);
        if (c == other->controls.end()) {
            if (controls.find(other->target) != controls.end()) {
                return other->IsPhase();
            }

            return (target != other->target) || (IsPhase() && other->IsPhase());
        }

        if (controls.find(other->target) != controls.end()) {
            return IsPhase() && other->IsPhase();
        }
        if (IsPhase()) {
            return true;
        }
        if (!IsPhaseInvert() ||
            !std::includes(other->controls.begin(), other->controls.end(), controls.begin(), controls.end())) {
            return false;
        }

        std::vector<bitCapInt> opfPows;
        opfPows.reserve(controls.size());
        for (const bitLenInt& ctrl : controls) {
            opfPows.emplace_back(pow2(std::distance(other->controls.begin(), other->controls.find(ctrl))));
        }
        const bitCapInt p = pow2(std::distance(other->controls.begin(), c));
        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : other->payloads) {
            bitCapInt pf = ZERO_BCI;
            for (size_t i = 0U; i < opfPows.size(); ++i) {
                if (bi_compare_0(payload.first & opfPows[i]) != 0) {
                    bi_or_ip(&pf, pow2(i));
                }
            }
            const auto& poi = payloads.find(pf);
            if ((poi == payloads.end()) || (norm(poi->second.get()[0]) > FP_NORM_EPSILON)) {
                nPayloads[payload.first] = payload.second;
            } else {
                nPayloads[payload.first ^ p] = payload.second;
            }
        }
        other->payloads = nPayloads;

        return true;
    }

    /**
     * To run as a uniformly controlled gate, generate my payload array.
     */
    std::unique_ptr<complex[]> MakeUniformlyControlledPayload()
    {
        const bitCapIntOcl maxQPower = pow2Ocl(controls.size());
        std::unique_ptr<complex[]> toRet(new complex[maxQPower << 2U]);
        QRACK_CONST complex identity[4] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
        for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
            complex* mtrx = toRet.get() + (i << 2U);
            const auto& p = payloads.find(i);
            if (p == payloads.end()) {
                std::copy(identity, identity + 4, mtrx);
                continue;
            }

            const complex* oMtrx = p->second.get();
            std::copy(oMtrx, oMtrx + 4U, mtrx);
        }

        return toRet;
    }

    /**
     * Convert my set of qubit indices to a vector
     */
    std::vector<bitLenInt> GetControlsVector() { return std::vector<bitLenInt>(controls.begin(), controls.end()); }

    /**
     * Erase a control index, if it exists, (via post selection).
     */
    void PostSelectControl(bitLenInt c, bool eigen)
    {
        const auto controlIt = controls.find(c);
        if (controlIt == controls.end()) {
            return;
        }

        const size_t cpos = std::distance(controls.begin(), controlIt);
        const bitCapInt midPow = pow2(cpos);
        bitCapInt lowMask = midPow;
        bi_decrement(&lowMask, 1U);
        const bitCapInt highMask = ~(lowMask | midPow);
        const bitCapInt qubitPow = pow2(cpos);
        const bitCapInt eigenPow = eigen ? qubitPow : ZERO_BCI;

        std::map<bitCapInt, std::shared_ptr<complex>> nPayloads;
        for (const auto& payload : payloads) {
            if (bi_compare(payload.first & qubitPow, eigenPow) != 0) {
                continue;
            }
            nPayloads.emplace((payload.first & lowMask) | ((payload.first & highMask) >> 1U), payload.second);
        }

        payloads = nPayloads;
        controls.erase(c);
    }
};

std::ostream& operator<<(std::ostream& os, const QCircuitGatePtr g);
std::istream& operator>>(std::istream& os, QCircuitGatePtr& g);

/**
 * Define and optimize a circuit, before running on a `QInterface`.
 */
class QCircuit;
typedef std::shared_ptr<QCircuit> QCircuitPtr;

class QCircuit {
protected:
    bool isCollapsed;
    bool isNearClifford;
    bitLenInt qubitCount;
    std::list<QCircuitGatePtr> gates;

public:
    /**
     * Default constructor
     */
    QCircuit(bool collapse = true, bool clifford = false)
        : isCollapsed(collapse)
        , isNearClifford(clifford)
        , qubitCount(0)
        , gates()
    {
        // Intentionally left blank
    }

    /**
     * Manual constructor
     */
    QCircuit(bitLenInt qbCount, const std::list<QCircuitGatePtr>& g, bool collapse = true, bool clifford = false)
        : isCollapsed(collapse)
        , isNearClifford(clifford)
        , qubitCount(qbCount)
    {
        for (const QCircuitGatePtr& gate : g) {
            gates.push_back(gate->Clone());
        }
    }

    QCircuitPtr Clone() { return std::make_shared<QCircuit>(qubitCount, gates, isCollapsed, isNearClifford); }

    QCircuitPtr Inverse()
    {
        QCircuitPtr clone = Clone();
        for (QCircuitGatePtr& gate : clone->gates) {
            for (auto& p : gate->payloads) {
                const complex* m = p.second.get();
                complex inv[4U]{ conj(m[0U]), conj(m[2U]), conj(m[1U]), conj(m[3U]) };
                std::copy(inv, inv + 4U, p.second.get());
            }
        }
        clone->gates.reverse();

        return clone;
    }

    /**
     * Get the (automatically calculated) count of qubits in this circuit, so far.
     */
    bitLenInt GetQubitCount() { return qubitCount; }

    /**
     * Set the count of qubits in this circuit, so far.
     */
    void SetQubitCount(bitLenInt n) { qubitCount = n; }

    /**
     * Return the raw list of gates.
     */
    std::list<QCircuitGatePtr> GetGateList() { return gates; }

    /**
     * Set the raw list of gates.
     */
    void SetGateList(std::list<QCircuitGatePtr> gl) { gates = gl; }

    /**
     * Add a `Swap` gate to the gate sequence.
     */
    void Swap(bitLenInt q1, bitLenInt q2)
    {
        if (q1 == q2) {
            return;
        }

        // If all swap gates are constructed in the same order, between high and low qubits, then the chances of
        // combining them might be higher.
        if (q1 > q2) {
            std::swap(q1, q2);
        }

        QRACK_CONST complex m[4] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
        const std::set<bitLenInt> s1 = { q1 };
        const std::set<bitLenInt> s2 = { q2 };
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, ONE_BCI));
        AppendGate(std::make_shared<QCircuitGate>(q2, m, s1, ONE_BCI));
        AppendGate(std::make_shared<QCircuitGate>(q1, m, s2, ONE_BCI));
    }

    /**
     * Append circuit (with identical qubit index mappings) at the end of this circuit.
     */
    void Append(QCircuitPtr circuit)
    {
        if (circuit->qubitCount > qubitCount) {
            qubitCount = circuit->qubitCount;
        }
        gates.insert(gates.end(), circuit->gates.begin(), circuit->gates.end());
    }

    /**
     * Combine circuit (with identical qubit index mappings) at the end of this circuit, by acting all additional
     * gates in sequence.
     */
    void Combine(QCircuitPtr circuit)
    {
        if (circuit->qubitCount > qubitCount) {
            qubitCount = circuit->qubitCount;
        }
        for (const QCircuitGatePtr& g : circuit->gates) {
            AppendGate(g);
        }
    }

    /**
     * Add a gate to the gate sequence.
     */
    void AppendGate(QCircuitGatePtr nGate);
    /**
     * Run this circuit.
     */
    void Run(QInterfacePtr qsim);

    /**
     * Check if an index is any target qubit of this circuit.
     */
    bool IsNonPhaseTarget(bitLenInt qubit)
    {
        for (const QCircuitGatePtr& gate : gates) {
            if ((gate->target == qubit) && !(gate->IsPhase())) {
                return true;
            }
        }

        return false;
    }

    /**
     * (If the qubit is not a target of a non-phase gate...) Delete this qubits' controls and phase targets.
     */
    void DeletePhaseTarget(bitLenInt qubit, bool eigen)
    {
        std::list<QCircuitGatePtr> nGates;
        gates.reverse();
        for (const QCircuitGatePtr& gate : gates) {
            if (gate->target == qubit) {
                continue;
            }
            QCircuitGatePtr nGate = gate->Clone();
            nGate->PostSelectControl(qubit, eigen);
            nGates.insert(nGates.begin(), nGate);
        }
        gates = nGates;
    }

    /**
     * Return (as a new QCircuit) just the gates on the past light cone of a set of qubit indices.
     */
    QCircuitPtr PastLightCone(std::set<bitLenInt>& qubits)
    {
        // We're working from latest gate to earliest gate.
        gates.reverse();

        std::list<QCircuitGatePtr> nGates;
        for (const QCircuitGatePtr& gate : gates) {
            // Is the target qubit on the light cone?
            if (qubits.find(gate->target) == qubits.end()) {
                // The target isn't on the light cone, but the controls might be.
                bool isNonCausal = true;
                for (const bitLenInt& c : gate->controls) {
                    if (qubits.find(c) != qubits.end()) {
                        isNonCausal = false;
                        break;
                    }
                }
                if (isNonCausal) {
                    // This gate is not on the past light cone.
                    continue;
                }
            }

            // This gate is on the past light cone.
            nGates.insert(nGates.begin(), gate->Clone());

            // Every qubit involved in this gate is now considered to be part of the past light cone.
            qubits.insert(gate->target);
            qubits.insert(gate->controls.begin(), gate->controls.end());
        }

        // Restore the original order of this QCircuit's gates.
        gates.reverse();

        return std::make_shared<QCircuit>(qubitCount, nGates, isCollapsed);
    }

#if ENABLE_ALU
    /** Add integer (without sign) */
    void INC(const bitCapInt& toAdd, bitLenInt start, bitLenInt length);
#endif
};

std::ostream& operator<<(std::ostream& os, const QCircuitPtr g);
std::istream& operator>>(std::istream& os, QCircuitPtr& g);
} //namespace Qrack

//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2024. All rights reserved.
//
// File: "qcircuit.cpp"
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qcircuit.hpp"

#include <iomanip>

namespace Qrack {

std::ostream& operator<<(std::ostream& os, const QCircuitGatePtr g)
{
    os << (size_t)g->target << " ";

    os << g->controls.size() << " ";
    for (const bitLenInt& c : g->controls) {
        os << (size_t)c << " ";
    }

    os << g->payloads.size() << " ";
#if FPPOW > 6
    os << std::setprecision(36);
#elif FPPOW > 5
    os << std::setprecision(17);
#endif
    for (const auto& p : g->payloads) {
        os << p.first << " ";
        for (size_t i = 0U; i < 4U; ++i) {
            os << p.second.get()[i] << " ";
        }
    }

    return os;
}

std::istream& operator>>(std::istream& is, QCircuitGatePtr& g)
{
    g->payloads.clear();

    size_t target;
    is >> target;
    g->target = (bitLenInt)target;

    size_t cSize;
    is >> cSize;
    for (size_t i = 0U; i < cSize; ++i) {
        size_t c;
        is >> c;
        g->controls.insert((bitLenInt)c);
    }

    size_t pSize;
    is >> pSize;
    for (size_t i = 0U; i < pSize; ++i) {
        bitCapInt k;
        is >> k;

        g->payloads[k] = std::shared_ptr<complex>(new complex[4], std::default_delete<complex[]>());
        for (size_t j = 0U; j < 4U; ++j) {
            is >> g->payloads[k].get()[j];
        }
    }

    return is;
}

std::ostream& operator<<(std::ostream& os, const QCircuitPtr c)
{
    os << (size_t)c->GetQubitCount() << " ";

    std::list<QCircuitGatePtr> gates = c->GetGateList();
    os << gates.size() << " ";
    for (const QCircuitGatePtr& g : gates) {
        os << g;
    }

    return os;
}

std::istream& operator>>(std::istream& is, QCircuitPtr& c)
{
    size_t qubitCount;
    is >> qubitCount;
    c->SetQubitCount((bitLenInt)qubitCount);

    size_t gSize;
    is >> gSize;
    std::list<QCircuitGatePtr> gl;
    for (size_t i = 0U; i < gSize; ++i) {
        QCircuitGatePtr g = std::make_shared<QCircuitGate>();
        is >> g;
        gl.push_back(g);
    }
    c->SetGateList(gl);

    return is;
}

void QCircuit::AppendGate(QCircuitGatePtr nGate)
{
    if (!isCollapsed) {
        gates.push_back(nGate);
        return;
    }

    if (nGate->IsIdentity()) {
        return;
    }

    if ((nGate->target + 1U) > qubitCount) {
        qubitCount = nGate->target + 1U;
    }
    if (!(nGate->controls.empty())) {
        const bitLenInt q = *(nGate->controls.rbegin());
        if ((q + 1U) > qubitCount) {
            qubitCount = (q + 1U);
        }
    }

    for (std::list<QCircuitGatePtr>::reverse_iterator gate = gates.rbegin(); gate != gates.rend(); ++gate) {
        if ((*gate)->TryCombine(nGate, isNearClifford)) {
            if ((*gate)->IsIdentity()) {
                std::list<QCircuitGatePtr>::reverse_iterator _gate = gate++;
                std::list<QCircuitGatePtr> head(_gate.base(), gates.end());
                gates.erase(gate.base(), gates.end());
                for (std::list<QCircuitGatePtr>::iterator g = head.begin(); g != head.end(); ++g) {
                    if (!nGate->CanCombine(*g, isNearClifford) && !nGate->CanPass(*g)) {
                        gates.push_back(*g);
                    } else {
                        AppendGate(*g);
                    }
                }
            }
            return;
        }
        if (!(*gate)->CanPass(nGate)) {
            gates.insert(gate.base(), { nGate });
            return;
        }
    }

    gates.push_front(nGate);
}

void QCircuit::Run(QInterfacePtr qsim)
{
    if (qsim->GetQubitCount() < qubitCount) {
        qsim->Allocate(qubitCount - qsim->GetQubitCount());
    }

    std::list<QCircuitGatePtr> nGates;
    if (gates.size() < 3U) {
        nGates = gates;
    } else {
        std::list<QCircuitGatePtr>::iterator end = gates.begin();
        std::advance(end, gates.size() - 2U);
        std::list<QCircuitGatePtr>::iterator gate;
        for (gate = gates.begin(); gate != end; ++gate) {
            if (!(*gate)->IsCnot()) {
                nGates.push_back(*gate);
                continue;
            }
            std::list<QCircuitGatePtr>::iterator adv = gate;
            ++adv;
            if (!(*adv)->IsCnot() || ((*adv)->target != *((*gate)->controls.begin())) ||
                ((*gate)->target != *((*adv)->controls.begin()))) {
                nGates.push_back(*gate);
                continue;
            }
            ++adv;
            if (!(*adv)->IsCnot() || ((*adv)->target != (*gate)->target) ||
                (*((*gate)->controls.begin()) != *((*adv)->controls.begin()))) {
                nGates.push_back(*gate);
                continue;
            }
            nGates.push_back(std::make_shared<QCircuitGate>((*gate)->target, *((*gate)->controls.begin())));
            gate = adv;
            if (std::distance(gate, gates.end()) < 3) {
                ++gate;
                break;
            }
        }
        for (; gate != gates.end(); ++gate) {
            nGates.push_back(*gate);
        }
    }

    for (const QCircuitGatePtr& gate : nGates) {
        const bitLenInt& t = gate->target;

        if (gate->controls.empty()) {
            qsim->Mtrx(gate->payloads[ZERO_BCI].get(), t);

            continue;
        }

        std::vector<bitLenInt> controls = gate->GetControlsVector();

        if (gate->payloads.empty()) {
            qsim->Swap(controls[0U], t);

            continue;
        }

        if (gate->payloads.size() == 1U) {
            const auto& payload = gate->payloads.begin();
            qsim->UCMtrx(controls, payload->second.get(), t, payload->first);

            continue;
        }

        std::unique_ptr<complex[]> payload = gate->MakeUniformlyControlledPayload();
        qsim->UniformlyControlledSingleBit(controls, t, payload.get());
    }
}
} // namespace Qrack

//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2024. All rights reserved.
//
// This section of the "qrack_misc.hpp" (pseudo-code) file is Qrack's very unique
// and important pareto-optimal approach to replacing Schmidt decomposition, for
// detecting separability, factorizing product states, and finding the closest
// possible separable state to an arbitrary entangled state. Notice that the
// "entanglement-breaking-channel" of `DecomposeDispose()` constructively aims to
// minimize the fidelity loss on a round trip through `Decompose()` and `Compose()`
// according to essentially the same principles as "Ordinary Least Squares" ("OLS").
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

namespace Qrack {

bool QInterface::TryDecompose(bitLenInt start, QInterfacePtr dest, real1_f error_tol)
{
    Finish();

    QInterfacePtr orig = Copy();
    orig->Decompose(start, dest);
    QInterfacePtr output = orig->Copy();
    orig->Compose(dest, start);

    const bool didSeparate = SumSqrDiff(orig) <= error_tol;

    if (didSeparate) {
        // The subsystem is separable.
        Copy(output);
    }

    return didSeparate;
}

/**
 * Minimally decompose a set of contigious bits from the separable unit. The
 * length of this separable unit is reduced by the length of bits decomposed, and
 * the bits removed are output in the destination QEngineCPU pointer. The
 * destination object must be initialized to the correct number of bits.
 */
void QEngineCPU::DecomposeDispose(bitLenInt start, bitLenInt length, QEngineCPUPtr destination)
{
    if (isBadBitRange(start, length, qubitCount)) {
        throw std::invalid_argument("QEngineCPU::DecomposeDispose range is out-of-bounds!");
    }

    if (!length) {
        return;
    }

    const bitLenInt nLength = qubitCount - length;

    if (!stateVec) {
        SetQubitCount(nLength);
        if (destination) {
            destination->ZeroAmplitudes();
        }
        return;
    }

    if (!nLength) {
        if (destination) {
            destination->stateVec = stateVec;
        }
        stateVec = NULL;
        SetQubitCount(0U);

        return;
    }

    if (destination && !destination->stateVec) {
        // Reinitialize stateVec RAM
        destination->SetPermutation(ZERO_BCI);
    }

    const bitCapIntOcl partPower = pow2Ocl(length);
    const bitCapIntOcl remainderPower = pow2Ocl(nLength);

    // Note that the extra parentheses mean to init as 0:
    std::unique_ptr<real1[]> remainderStateProb(new real1[remainderPower]());
    std::unique_ptr<real1[]> remainderStateAngle(new real1[remainderPower]());
    std::unique_ptr<real1[]> partStateProb;
    std::unique_ptr<real1[]> partStateAngle;
    if (destination) {
        partStateProb = std::unique_ptr<real1[]>(new real1[partPower]());
        partStateAngle = std::unique_ptr<real1[]>(new real1[partPower]());
    }

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (destination) {
        par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j = lcv & pow2MaskOcl(start);
            j |= (lcv ^ j) << length;

            for (bitCapIntOcl k = 0U; k < partPower; ++k) {
                const complex amp = stateVec->read(j | (k << start));
                const real1 nrm = norm(amp);
                remainderStateProb[lcv] += nrm;
                if (nrm > amplitudeFloor) {
                    partStateAngle[k] += arg(amp) * nrm;
                }
            }
        });
        par_for(0U, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const bitCapIntOcl startMask = pow2MaskOcl(start);
            const bitCapIntOcl j = lcv << start;

            for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
                bitCapIntOcl l = k & startMask;
                l |= j | ((k ^ l) << length);

                const complex amp = stateVec->read(l);
                const real1 nrm = norm(amp);
                partStateProb[lcv] += nrm;
                if (nrm > amplitudeFloor) {
                    remainderStateAngle[k] += arg(amp) * nrm;
                }
            }

            const real1 prob = partStateProb[lcv];
            if (prob > amplitudeFloor) {
                partStateAngle[lcv] /= prob;
            }
        });
        par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const real1 prob = remainderStateProb[lcv];
            if (prob > amplitudeFloor) {
                remainderStateAngle[lcv] /= prob;
            }
        });
    } else {
        par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            bitCapIntOcl j = lcv & pow2MaskOcl(start);
            j |= (lcv ^ j) << length;

            for (bitCapIntOcl k = 0U; k < partPower; ++k) {
                remainderStateProb[lcv] += norm(stateVec->read(j | (k << start)));
            }
        });
        par_for(0U, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const bitCapIntOcl startMask = pow2MaskOcl(start);
            const bitCapIntOcl j = lcv << start;

            for (bitCapIntOcl k = 0U; k < remainderPower; ++k) {
                bitCapIntOcl l = k & startMask;
                l |= j | ((k ^ l) << length);

                const complex amp = stateVec->read(l);
                const real1 nrm = norm(amp);
                if (nrm > amplitudeFloor) {
                    remainderStateAngle[k] += arg(amp) * nrm;
                }
            }
        });
        par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            const real1 prob = remainderStateProb[lcv];
            if (prob > amplitudeFloor) {
                remainderStateAngle[lcv] /= prob;
            }
        });
    }

    if (destination) {
        destination->Dump();

        par_for(0U, partPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
            destination->stateVec->write(lcv,
                (real1)(std::sqrt((real1_s)partStateProb[lcv])) *
                    complex(cos(partStateAngle[lcv]), sin(partStateAngle[lcv])));
        });

        partStateProb.reset();
        partStateAngle.reset();
    }

    SetQubitCount(nLength);

    ResetStateVec(AllocStateVec(maxQPowerOcl));

    par_for(0U, remainderPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        stateVec->write(lcv,
            (real1)(std::sqrt((real1_s)remainderStateProb[lcv])) *
                complex(cos(remainderStateAngle[lcv]), sin(remainderStateAngle[lcv])));
    });
}

void QEngineCPU::Decompose(bitLenInt start, QInterfacePtr destination)
{
    DecomposeDispose(start, destination->GetQubitCount(), std::dynamic_pointer_cast<QEngineCPU>(destination));
}

/**
 * Combine (a copy of) another QEngineCPU with this one, inserted at the "start" index.
 * (This is just a "Kronecker product" or "tensor product.")
 */
bitLenInt QEngineCPU::Compose(QEngineCPUPtr toCopy, bitLenInt start)
{
    if (start > qubitCount) {
        throw std::invalid_argument("QEngineCPU::Compose start index is out-of-bounds!");
    }

    if (!qubitCount) {
        Compose(toCopy);
        return 0U;
    }

    if (!toCopy->qubitCount) {
        return qubitCount;
    }

    const bitLenInt nQubitCount = qubitCount + toCopy->qubitCount;

    if (nQubitCount > QRACK_MAX_CPU_QB_DEFAULT) {
        throw std::invalid_argument(
            "Cannot instantiate a QEngineCPU with greater capacity than environment variable QRACK_MAX_CPU_QB.");
    }

    if (!stateVec || !toCopy->stateVec) {
        // Compose will have a wider but 0 stateVec
        ZeroAmplitudes();
        SetQubitCount(nQubitCount);
        return start;
    }

    const bitLenInt oQubitCount = toCopy->qubitCount;
    const bitCapIntOcl nMaxQPower = pow2Ocl(nQubitCount);
    const bitCapIntOcl startMask = pow2MaskOcl(start);
    const bitCapIntOcl midMask = bitRegMaskOcl(start, oQubitCount);
    const bitCapIntOcl endMask = pow2MaskOcl(qubitCount + oQubitCount) & ~(startMask | midMask);

    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (toCopy->doNormalize) {
        toCopy->NormalizeState();
    }
    toCopy->Finish();

    StateVectorPtr nStateVec = AllocStateVec(nMaxQPower);

    par_for(0U, nMaxQPower, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        nStateVec->write(lcv,
            stateVec->read((lcv & startMask) | ((lcv & endMask) >> oQubitCount)) *
                toCopy->stateVec->read((lcv & midMask) >> start));
    });

    SetQubitCount(nQubitCount);

    ResetStateVec(nStateVec);

    return start;
}

/**
 * Find the difference in L2 norm projection between "this" and "toCompare."
 * (This is how fidelity is usually calculated in quantum mechanics, though
 * we return 1 - fidelity from this `SumSqrDiff()` method.)
 */
real1_f QEngineCPU::SumSqrDiff(QEngineCPUPtr toCompare)
{
    if (!toCompare) {
        return ONE_R1_F;
    }

    if (this == toCompare.get()) {
        return ZERO_R1_F;
    }

    // If the qubit counts are unequal, these can't be approximately equal objects.
    if (qubitCount != toCompare->qubitCount) {
        // Max square difference:
        return ONE_R1_F;
    }

    // Make sure both engines are normalized
    if (doNormalize) {
        NormalizeState();
    }
    Finish();

    if (toCompare->doNormalize) {
        toCompare->NormalizeState();
    }
    toCompare->Finish();

    if (!stateVec && !toCompare->stateVec) {
        return ZERO_R1_F;
    }

    if (!stateVec) {
        toCompare->UpdateRunningNorm();
        return (real1_f)(toCompare->runningNorm);
    }

    if (!toCompare->stateVec) {
        UpdateRunningNorm();
        return (real1_f)runningNorm;
    }

    const unsigned numCores = GetConcurrencyLevel();
    std::unique_ptr<complex[]> partInner(new complex[numCores]());

    par_for(0U, maxQPowerOcl, [&](const bitCapIntOcl& lcv, const unsigned& cpu) {
        partInner[cpu] += conj(stateVec->read(lcv)) * toCompare->stateVec->read(lcv);
    });

    complex totInner = ZERO_CMPLX;
    for (unsigned i = 0U; i < numCores; ++i) {
        totInner += partInner[i];
    }

    return ONE_R1_F - clampProb((real1_f)norm(totInner));
}

} // namespace Qrack

//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2024. All rights reserved.
//
// The remainder of "qrack_misc.hpp" is a snippet from our near-Clifford technique.
//
// There was an open-access PRX article that pointed out, at least in an appendix,
// that it's always possible to encode general non-Clifford variational phase gates on
// ancillary stabilizer qubits, rotating the basis of ancilla measurement, so that a
// post-selected |0> qubit measurement outcome on the ancilla will succeed in
// "injecting" the non-Clifford gate _without any semi-classical "correction" gate,_ ever.
//
// In simulo, it turns out this deferment of gadget injection (with post-selection)
// only gets as far as purely unitary simulation; once we measure logical qubits, we
// can no longer proceed in general purely via stabilizer states with this special
// type of post-selected gadgets. However, Qrack's stroke of insight was that an
// auxiliary state vector simulation on a set of amplitudes with exponential Hilbert
// space dimensionality of _just_ the ancillary qubits can be used to calculate one
// near-Clifford logical amplitude at a time. So, for measurement shots, which require
// us to calculate 2^n amplitudes for n logical qubits, the cost is exponential in
// both the number of ancillary qubits and, separately, the number of logical qubits.
// However, it is entirely tractable within memory limits if I can hold 2^m complex
// amplitudes in memory for m count of non-Clifford gates in total, though, and it
// encompasses _all variational phase gates._ (Plus, we're not naive about optimizing
// away the need for additional ancillae where we can avoid them, per usual Qrack
// implementation standards.)
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

namespace Qrack {

complex QStabilizerHybrid::GetAmplitudeOrProb(const bitCapInt& perm, bool isProb)
{
    if (engine) {
        return engine->GetAmplitude(perm);
    }

#if ENABLE_ENV_VARS
    if (!isRoundingFlushed && getenv("QRACK_NONCLIFFORD_ROUNDING_THRESHOLD")) {
        roundingThreshold = (real1_f)std::stof(std::string(getenv("QRACK_NONCLIFFORD_ROUNDING_THRESHOLD")));
    }
#endif
    const bool isRounded = roundingThreshold > FP_NORM_EPSILON;
    const QUnitCliffordPtr origStabilizer =
        isRounded ? std::dynamic_pointer_cast<QUnitClifford>(stabilizer->Clone()) : NULL;
    const bitLenInt origAncillaCount = ancillaCount;
    const bitLenInt origDeadAncillaCount = deadAncillaCount;
    std::vector<MpsShardPtr> origShards = isRounded ? shards : std::vector<MpsShardPtr>();
    if (isRounded) {
        for (size_t i = 0U; i < origShards.size(); ++i) {
            if (origShards[i]) {
                origShards[i] = origShards[i]->Clone();
            }
        }
        RdmCloneFlush(roundingThreshold);
    }

    if ((isProb && !ancillaCount && !IsLogicalProbBuffered()) || !IsBuffered()) {
        const complex toRet = stabilizer->GetAmplitude(perm);

        if (isRounded) {
            stabilizer = origStabilizer;
            ancillaCount = origAncillaCount;
            deadAncillaCount = origDeadAncillaCount;
            shards = origShards;
        }

        return toRet;
    }

    std::vector<bitLenInt> indices;
    std::vector<bitCapInt> perms{ perm };
    for (bitLenInt i = 0U; i < qubitCount; ++i) {
        if (!shards[i]) {
            continue;
        }
        indices.push_back(i);
        perms.push_back(perm ^ pow2(i));
    }

    if (!ancillaCount) {
        std::vector<complex> amps;
        amps.reserve(perms.size());
        if (stateMapCache) {
            for (size_t i = 0U; i < perms.size(); ++i) {
                amps.push_back(stateMapCache->get(perms[i]));
            }
        } else {
            amps = stabilizer->GetAmplitudes(perms);
        }
        complex amp = amps[0U];
        for (size_t i = 1U; i < amps.size(); ++i) {
            const bitLenInt j = indices[i - 1U];
            const complex* mtrx = shards[j]->gate;
            if (bi_and_1(perm >> j)) {
                amp = mtrx[2U] * amps[i] + mtrx[3U] * amp;
            } else {
                amp = mtrx[0U] * amp + mtrx[1U] * amps[i];
            }
        }

        if (isRounded) {
            stabilizer = origStabilizer;
            ancillaCount = origAncillaCount;
            deadAncillaCount = origDeadAncillaCount;
            shards = origShards;
        }

        return amp;
    }

    const bitLenInt aStride = indices.size() + 1U;
    const bitCapIntOcl ancillaPow = pow2Ocl(ancillaCount);
    for (bitCapIntOcl i = 1U; i < ancillaPow; ++i) {
        const bitCapInt ancillaPerm = i << qubitCount;
        for (size_t j = 0U; j < aStride; ++j) {
            perms.push_back(perms[j] | ancillaPerm);
        }
    }

    std::vector<complex> amps;
    amps.reserve(perms.size());
    if (stateMapCache) {
        for (size_t i = 0U; i < perms.size(); ++i) {
            amps.push_back(stateMapCache->get(perms[i]));
        }
    } else {
        amps = stabilizer->GetAmplitudes(perms);
    }

    std::vector<QInterfaceEngine> et = engineTypes;
    for (int i = et.size() - 1U; i >= 0; --i) {
        if ((et[i] == QINTERFACE_BDT_HYBRID) || (et[i] == QINTERFACE_BDT)) {
            et.erase(et.begin() + i);
        }
    }
    if (et.empty()) {
        et.push_back(QINTERFACE_OPTIMAL_BASE);
    }
    QEnginePtr aEngine = std::dynamic_pointer_cast<QEngine>(
        CreateQuantumInterface(et, ancillaCount, ZERO_BCI, rand_generator, ONE_CMPLX, false, false, useHostRam, devID,
            useRDRAND, false, (real1_f)amplitudeFloor, deviceIDs, thresholdQubits, separabilityThreshold));

#if ENABLE_COMPLEX_X2
    std::vector<complex2> top;
    std::vector<complex2> bottom;
    // For variable scoping, only:
    if (true) {
        complex amp = amps[0U];
        for (bitLenInt i = 1U; i < aStride; ++i) {
            const bitLenInt j = indices[i - 1U];
            const complex* mtrx = shards[j]->gate;
            top.emplace_back(mtrx[0U], mtrx[1U]);
            bottom.emplace_back(mtrx[3U], mtrx[2U]);
            complex2 amp2(amp, amps[i]);
            if (bi_and_1(perm >> j)) {
                amp2 = amp2 * bottom.back();
            } else {
                amp2 = amp2 * top.back();
            }
            amp = amp2.c(0U) + amp2.c(1U);
        }
        aEngine->SetAmplitude(0U, amp);
    }
    for (bitCapIntOcl a = 1U; a < ancillaPow; ++a) {
        const bitCapIntOcl offset = a * aStride;
        complex amp = amps[offset];
        for (bitLenInt i = 1U; i < aStride; ++i) {
            const bitLenInt j = indices[i - 1U];
            complex2 amp2(amp, amps[i]);
            if (bi_and_1(perm >> j)) {
                amp2 = amp2 * bottom[j];
            } else {
                amp2 = amp2 * top[j];
            }
            amp = amp2.c(0U) + amp2.c(1U);
        }
        aEngine->SetAmplitude(a, amp);
    }
    top.clear();
    bottom.clear();
#else
    for (bitCapIntOcl a = 0U; a < ancillaPow; ++a) {
        const bitCapIntOcl offset = a * aStride;
        complex amp = amps[offset];
        for (bitLenInt i = 1U; i < aStride; ++i) {
            const bitLenInt j = indices[i - 1U];
            const complex* mtrx = shards[j]->gate;
            const complex oAmp = amps[i + offset];
            if (bi_and_1(perm >> j)) {
                amp = mtrx[3U] * amp + mtrx[2U] * oAmp;
            } else {
                amp = mtrx[0U] * amp + mtrx[1U] * oAmp;
            }
        }
        aEngine->SetAmplitude(a, amp);
    }
#endif
    amps.clear();

    for (bitLenInt i = 0U; i < ancillaCount; ++i) {
        const MpsShardPtr& shard = shards[i + qubitCount];
        if (shard) {
            aEngine->Mtrx(shard->gate, i);
        }
    }

    if (isRounded) {
        stabilizer = origStabilizer;
        ancillaCount = origAncillaCount;
        deadAncillaCount = origDeadAncillaCount;
        shards = origShards;
    }

    return (real1)pow(SQRT2_R1, (real1)ancillaCount) * aEngine->GetAmplitude(ZERO_BCI);
}

} // namespace Qrack
