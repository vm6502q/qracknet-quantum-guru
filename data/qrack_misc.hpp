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
        : target(0U)
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
        payloads[ZERO_BCI] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4U, payloads[ZERO_BCI].get());
    }

    /**
     * Controlled gate constructor
     */
    QCircuitGate(bitLenInt trgt, const complex matrix[], const std::set<bitLenInt>& ctrls, const bitCapInt& perm)
        : target(trgt)
        , controls(ctrls)
    {
        const std::shared_ptr<complex>& p = payloads[perm] =
            std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        std::copy(matrix, matrix + 4U, p.get());
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
            payloads[payload.first] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
            std::copy(payload.second.get(), payload.second.get() + 4U, payloads[payload.first].get());
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

        if (clifford) {
            if (controls.empty() && other->controls.empty()) {
                return true;
            }

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

        if (controls.empty() || other->controls.empty()) {
            return true;
        }

        return std::includes(other->controls.begin(), other->controls.end(), controls.begin(), controls.end()) ||
            std::includes(controls.begin(), controls.end(), other->controls.begin(), other->controls.end());
    }

    /**
     * Set this gate to the identity operator.
     */
    void Clear()
    {
        controls.clear();
        payloads.clear();

        payloads[ZERO_BCI] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
        complex* p = payloads[ZERO_BCI].get();
        p[0U] = ONE_CMPLX;
        p[1U] = ZERO_CMPLX;
        p[2U] = ZERO_CMPLX;
        p[3U] = ONE_CMPLX;
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

            std::shared_ptr<complex> np = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
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
            if (amp_leq_0(l[0U] - h[0U]) && amp_leq_0(l[1U] - h[1U]) && amp_leq_0(l[2U] - h[2U]) &&
                amp_leq_0(l[3U] - h[3U])) {
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
                    std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
                std::copy(payload.second.get(), payload.second.get() + 4U, p.get());

                continue;
            }

            complex* p = pit->second.get();
            complex out[4];
            mul2x2(payload.second.get(), p, out);

            if (amp_leq_0(out[1U]) && amp_leq_0(out[2U]) && amp_leq_0(ONE_CMPLX - out[0U]) &&
                amp_leq_0(ONE_CMPLX - out[3U])) {
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
        const bitCapInt controlPow = pow2(controls.size());
        if (payloads.size() == controlPow) {
            const complex* refP = payloads.begin()->second.get();
            if (norm(refP[0U] - refP[3U]) > FP_NORM_EPSILON) {
                return false;
            }
            const complex phaseFac = refP[0U];
            for (const auto& payload : payloads) {
                complex* p = payload.second.get();
                if ((norm(p[1U]) > FP_NORM_EPSILON) || (norm(p[2U]) > FP_NORM_EPSILON) ||
                    (norm(phaseFac - p[0U]) > FP_NORM_EPSILON) || (norm(phaseFac - p[3U]) > FP_NORM_EPSILON)) {
                    return false;
                }
            }
            return true;
        }

        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1U]) > FP_NORM_EPSILON) || (norm(p[2U]) > FP_NORM_EPSILON) ||
                (norm(ONE_CMPLX - p[0U]) > FP_NORM_EPSILON) || (norm(ONE_CMPLX - p[3U]) > FP_NORM_EPSILON)) {
                return false;
            }
        }

        return true;
    }

    /**
     * Am I a phase gate?
     */
    bool IsPhase()
    {
        for (const auto& payload : payloads) {
            complex* p = payload.second.get();
            if ((norm(p[1U]) > FP_NORM_EPSILON) || (norm(p[2U]) > FP_NORM_EPSILON)) {
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
            if ((norm(p[0U]) > FP_NORM_EPSILON) || (norm(p[3U]) > FP_NORM_EPSILON)) {
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
     * Am I a CNOT gate?
     */
    bool IsAntiCnot()
    {
        if ((controls.size() != 1U) || (payloads.size() != 1U) || (payloads.find(ZERO_BCI) == payloads.end())) {
            return false;
        }
        complex* p = payloads[ZERO_BCI].get();
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
            if ((poi == payloads.end()) || (norm(poi->second.get()[0U]) > FP_NORM_EPSILON)) {
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
        QRACK_CONST complex identity[4U] = { ONE_CMPLX, ZERO_CMPLX, ZERO_CMPLX, ONE_CMPLX };
        for (bitCapIntOcl i = 0U; i < maxQPower; ++i) {
            complex* mtrx = toRet.get() + (i << 2U);
            const auto& p = payloads.find(i);
            if (p == payloads.end()) {
                std::copy(identity, identity + 4U, mtrx);
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
        , qubitCount(0U)
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

        QRACK_CONST complex m[4U] = { ZERO_CMPLX, ONE_CMPLX, ONE_CMPLX, ZERO_CMPLX };
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
    bool AppendGate(QCircuitGatePtr nGate);
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
} // namespace Qrack


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

namespace Qrack {

std::ostream& operator<<(std::ostream& os, const QCircuitGatePtr g)
{
    os << (size_t)g->target << " ";

    os << g->controls.size() << " ";
    for (const bitLenInt& c : g->controls) {
        os << (size_t)c << " ";
    }

    os << g->payloads.size() << " ";
    for (const auto& p : g->payloads) {
        os << (size_t)p.first << " ";
        const complex* mtrx = p.second.get();
        for (size_t i = 0U; i < 4U; ++i) {
            os << mtrx[i] << " ";
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

        g->payloads[k] = std::shared_ptr<complex>(new complex[4U], std::default_delete<complex[]>());
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

bool QCircuit::AppendGate(QCircuitGatePtr nGate)
{
    if (!isCollapsed) {
        gates.push_back(nGate);
        return false;
    }

    if (nGate->IsIdentity()) {
        return true;
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

    std::set<bitLenInt> nQubits(nGate->controls);
    nQubits.insert(nGate->target);
    bool didCommute = false;
    for (std::list<QCircuitGatePtr>::reverse_iterator gate = gates.rbegin(); gate != gates.rend(); ++gate) {
        if ((*gate)->TryCombine(nGate, isNearClifford)) {
            if ((*gate)->IsIdentity()) {
                std::set<bitLenInt> gQubits((*gate)->controls);
                gQubits.insert((*gate)->target);
                std::list<QCircuitGatePtr>::reverse_iterator _gate = gate++;
                std::list<QCircuitGatePtr> head(_gate.base(), gates.end());
                gates.erase(gate.base(), gates.end());
                for (; head.size() && gQubits.size(); head.erase(head.begin())) {
                    std::set<bitLenInt> hQubits(head.front()->controls);
                    hQubits.insert((*head.begin())->target);
                    if (!std::any_of(hQubits.begin(), hQubits.end(),
                            [&gQubits](bitLenInt element) { return gQubits.count(element) > 0; })) {
                        gates.push_back(head.front());
                        continue;
                    }
                    if (AppendGate(head.front())) {
                        gQubits.insert(hQubits.begin(), hQubits.end());
                    } else {
                        for (const auto& hq : hQubits) {
                            gQubits.erase(hq);
                        }
                    }
                }
                gates.insert(gates.end(), head.begin(), head.end());
            }
            return true;
        }
        if (!(*gate)->CanPass(nGate)) {
            gates.insert(gate.base(), { nGate });
            return didCommute;
        }
        std::set<bitLenInt> gQubits((*gate)->controls);
        gQubits.insert((*gate)->target);
        didCommute |= std::any_of(
            nQubits.begin(), nQubits.end(), [&gQubits](bitLenInt element) { return gQubits.count(element) > 0; });
    }

    gates.push_front(nGate);

    return didCommute;
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
            bool isAnti = (*gate)->IsAntiCnot();
            if (!((*gate)->IsCnot() || isAnti))
            {
                nGates.push_back(*gate);
                continue;
            }
            std::list<QCircuitGatePtr>::iterator adv = gate;
            ++adv;
            if (!((isAnti && (*adv)->IsAntiCnot()) || (!isAnti && (*adv)->IsCnot())) ||
                ((*adv)->target != *((*gate)->controls.begin())) || ((*gate)->target != *((*adv)->controls.begin()))) {
                nGates.push_back(*gate);
                continue;
            }
            ++adv;
            if (!((isAnti && (*adv)->IsAntiCnot()) || (!isAnti && (*adv)->IsCnot())) ||
                ((*adv)->target != (*gate)->target) || (*((*gate)->controls.begin()) != *((*adv)->controls.begin()))) {
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

/**
 * From "quantum binary decision diagrams" (QBDD) or "quantum binary decision trees" (`QBdt`, hence),
 * the "separability problem" if effectively solved _"structurally"_ for all potential bipartite
 * boundaries between an aligned low-index qubit subsytem and an aligned high-index qubit subsystem.
 * Empirically, this check can take on order of just a millisecond for a large composite system of
 * GHZ subsystems.
 */
bool QBdt::IsSeparable(bitLenInt start)
{
    if (!start || (start >= qubitCount)) {
        throw std::invalid_argument(
            "QBdt::IsSeparable() start parameter must be at least 1 and less than the QBdt qubit width!");
    }

    // If the tree has been fully reduced, this should ALWAYS be the same for ALL branches
    // (that have nonzero amplitude), if-and-only-if the state is separable.
    QBdtNodeInterfacePtr subsystemPtr = NULL;

    const bitCapInt qPower = pow2(start);
    bool result = true;

    par_for_qbdt(
        qPower, start,
        [this, start, &subsystemPtr, &result](const bitCapInt& i) {
            QBdtNodeInterfacePtr leaf = root;
            for (bitLenInt j = 0U; j < start; ++j) {
                leaf = leaf->branches[SelectBit(i, start - (j + 1U))];
                if (!leaf) {
                    // The immediate parent of "leaf" has 0 amplitude.
                    return (bitCapInt)(pow2(start - j) - ONE_BCI);
                }
            }

            if (!leaf->branches[0U] || !leaf->branches[1U]) {
                // "leaf" is a 0-amplitude branch.
                return ZERO_BCI;
            }

            // "leaf" is nonzero.
            // Every such instance must be identical.

            if (!subsystemPtr) {
                // Even if another thread "clobbers" this assignment,
                // then the equality check afterward will fail.
                subsystemPtr = leaf;
            }

            if (subsystemPtr != leaf) {
                // There are at least two distinct possible subsystem states for the "high-index" subsystem,
                // depending specifically on which dimension of the "low-index" subsystem we're inspecting.
                result = false;
                return (bitCapInt)(pow2(start) - ONE_BCI);
            }

            return ZERO_BCI;
        },
        false);

    return result;
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2024. All rights reserved.
//
// "A quantum-inspired Monte Carlo integer factoring algorithm"
//
// This library was originally called ["Qimcifa"](https://github.com/vm6502q/qimcifa) and demonstrated a (Shor's-like) "quantum-inspired" algorithm for integer factoring. It has
// since been developed into a general factoring algorithm and tool.
//
// `FindAFactor` uses heavily wheel-factorized brute-force "exhaust" numbers as "smooth" inputs to Quadratic Sieve, widely regarded as the asymptotically second fastest algorithm
// class known for cryptographically relevant semiprime factoring. `FindAFactor` is C++ based, with `pybind11`, which tends to make it faster than pure Python approaches. For the
// quick-and-dirty application of finding _any single_ nontrivial factor, something like at least 80% of positive integers will factorize in a fraction of a second, but the most
// interesting cases to consider are semiprime numbers, for which `FindAFactor` should be about as asymptotically competitive as similar Quadratic Sieve implementations.
//
// Our original contribution to Quadratic Sieve seems to be wheel factorization to 13 or 17 and maybe the idea of using the "exhaust" of a brute-force search for smooth number
// inputs for Quadratic Sieve. For wheel factorization (or "gear factorization"), we collect a short list of the first primes and remove all of their multiples from a "brute-force"
// guessing range by mapping a dense contiguous integer set, to a set without these multiples, relying on both a traditional "wheel," up to a middle prime number (of `11`), and a
// "gear-box" that stores increment values per prime according to the principles of wheel factorization, but operating semi-independently, to reduce space of storing the full
// wheel.
//
// Beyond this, we gain a functional advantage of a square-root over a more naive approach, by setting the brute force guessing range only between the highest prime in wheel
// factorization and the (modular) square root of the number to factor: if the number is semiprime, there is exactly one correct answer in this range, but including both factors in
// the range to search would cost us the square root advantage.
//
// Factoring this way is surprisingly easy to distribute: basically 0 network communication is needed to coordinate an arbitrarily high amount of parallelism to factor a single
// number. Each brute-force trial division instance is effectively 100% independent of all others (i.e. entirely "embarrassingly parallel"), and these guesses can seed independent
// Gaussian elimination matrices, so `FindAFactor` offers an extremely simply interface that allows work to be split between an arbitrarily high number of nodes with absolutely no
// network communication at all. In terms of incentives of those running different, cooperating nodes in the context of this specific number of integer factoring, all one
// ultimately cares about is knowing the correct factorization answer _by any means._ For pratical applications, there is no point at all in factoring a number whose factors are
// already known. When a hypothetical answer is forwarded to the (0-communication) "network" of collaborating nodes, _it is trivial to check whether the answer is correct_ (such as
// by simply entering the multiplication and equality check with the original number into a Python shell console)! Hence, collaborating node operators only need to trust that all
// participants in the "network" are actually performing their alloted segment of guesses and would actually communicate the correct answer to the entire group of collaborating
// nodes if any specific invidual happened to find the answer, but any purported answer is still trivial to verify.
//
//**Special thanks to OpenAI GPT "Elara," for indicated region of contributed code!**
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "dispatchqueue.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <float.h>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdlib.h>
#include <string>
#include <time.h>

#include <boost/dynamic_bitset.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace Qimcifa {

typedef boost::multiprecision::cpp_int BigInteger;

enum Wheel { ERROR = 0, WHEEL1 = 1, WHEEL2 = 2, WHEEL3 = 6, WHEEL5 = 30, WHEEL7 = 210, WHEEL11 = 2310 };

Wheel wheelByPrimeCardinal(int i) {
  switch (i) {
  case 0:
    return WHEEL1;
  case 1:
    return WHEEL2;
  case 2:
    return WHEEL3;
  case 3:
    return WHEEL5;
  case 4:
    return WHEEL7;
  case 5:
    return WHEEL11;
  default:
    return ERROR;
  }
}

// See https://stackoverflow.com/questions/101439/the-most-efficient-way-to-implement-an-integer-based-power-function-powint-int
BigInteger ipow(BigInteger base, unsigned exp) {
  BigInteger result = 1U;
  for (;;) {
    if (exp & 1U) {
      result *= base;
    }
    exp >>= 1U;
    if (!exp) {
      break;
    }
    base *= base;
  }

  return result;
}

inline size_t log2(BigInteger n) {
  size_t pow = 0U;
  while (n >>= 1U) {
    ++pow;
  }
  return pow;
}

inline BigInteger gcd(const BigInteger& n1, const BigInteger& n2) {
  if(!n2) {
    return n1;
  }
  return gcd(n2, n1 % n2);
}

BigInteger sqrt(const BigInteger &toTest) {
  // Otherwise, find b = sqrt(b^2).
  BigInteger start = 1U, end = toTest >> 1U, ans = 0U;
  do {
    const BigInteger mid = (start + end) >> 1U;

    // If toTest is a perfect square
    const BigInteger sqr = mid * mid;
    if (sqr == toTest) {
      return mid;
    }

    if (sqr < toTest) {
      // Since we need floor, we update answer when mid*mid is smaller than p, and move closer to sqrt(p).
      start = mid + 1U;
      ans = mid;
    } else {
      // If mid*mid is greater than p
      end = mid - 1U;
    }
  } while (start <= end);

  return ans;
}

// We are multiplying out the first distinct primes, below.

// Make this NOT a multiple of 2.
inline BigInteger forward2(const size_t &p) { return (p << 1U) | 1U; }

inline size_t backward2(const BigInteger &p) { return (size_t)(p >> 1U); }

// Make this NOT a multiple of 2 or 3.
inline BigInteger forward3(const size_t &p) { return (p << 1U) + (~(~p | 1U)) - 1U; }

inline size_t backward3(const BigInteger &n) { return (size_t)((~(~n | 1U)) / 3U) + 1U; }

constexpr unsigned char wheel5[8U] = {1U, 7U, 11U, 13U, 17U, 19U, 23U, 29U};

// Make this NOT a multiple of 2, 3, or 5.
BigInteger forward5(const size_t &p) { return wheel5[p % 8U] + (p / 8U) * 30U; }

size_t backward5(const BigInteger &n) { return std::distance(wheel5, std::lower_bound(wheel5, wheel5 + 8U, (size_t)(n % 30U))) + 8U * (size_t)(n / 30U) + 1U; }

constexpr unsigned char wheel7[48U] = {1U,   11U,  13U,  17U,  19U,  23U,  29U,  31U,  37U,  41U,  43U,  47U,  53U,  59U,  61U,  67U,
                                       71U,  73U,  79U,  83U,  89U,  97U,  101U, 103U, 107U, 109U, 113U, 121U, 127U, 131U, 137U, 139U,
                                       143U, 149U, 151U, 157U, 163U, 167U, 169U, 173U, 179U, 181U, 187U, 191U, 193U, 197U, 199U, 209U};

// Make this NOT a multiple of 2, 3, 5, or 7.
BigInteger forward7(const size_t &p) { return wheel7[p % 48U] + (p / 48U) * 210U; }

size_t backward7(const BigInteger &n) { return std::distance(wheel7, std::lower_bound(wheel7, wheel7 + 48U, (size_t)(n % 210U))) + 48U * (size_t)(n / 210U) + 1U; }

constexpr unsigned short wheel11[480U] = {
    1U,    13U,   17U,   19U,   23U,   29U,   31U,   37U,   41U,   43U,   47U,   53U,   59U,   61U,   67U,   71U,   73U,   79U,   83U,   89U,   97U,   101U,  103U,  107U,
    109U,  113U,  127U,  131U,  137U,  139U,  149U,  151U,  157U,  163U,  167U,  169U,  173U,  179U,  181U,  191U,  193U,  197U,  199U,  211U,  221U,  223U,  227U,  229U,
    233U,  239U,  241U,  247U,  251U,  257U,  263U,  269U,  271U,  277U,  281U,  283U,  289U,  293U,  299U,  307U,  311U,  313U,  317U,  323U,  331U,  337U,  347U,  349U,
    353U,  359U,  361U,  367U,  373U,  377U,  379U,  383U,  389U,  391U,  397U,  401U,  403U,  409U,  419U,  421U,  431U,  433U,  437U,  439U,  443U,  449U,  457U,  461U,
    463U,  467U,  479U,  481U,  487U,  491U,  493U,  499U,  503U,  509U,  521U,  523U,  527U,  529U,  533U,  541U,  547U,  551U,  557U,  559U,  563U,  569U,  571U,  577U,
    587U,  589U,  593U,  599U,  601U,  607U,  611U,  613U,  617U,  619U,  629U,  631U,  641U,  643U,  647U,  653U,  659U,  661U,  667U,  673U,  677U,  683U,  689U,  691U,
    697U,  701U,  703U,  709U,  713U,  719U,  727U,  731U,  733U,  739U,  743U,  751U,  757U,  761U,  767U,  769U,  773U,  779U,  787U,  793U,  797U,  799U,  809U,  811U,
    817U,  821U,  823U,  827U,  829U,  839U,  841U,  851U,  853U,  857U,  859U,  863U,  871U,  877U,  881U,  883U,  887U,  893U,  899U,  901U,  907U,  911U,  919U,  923U,
    929U,  937U,  941U,  943U,  947U,  949U,  953U,  961U,  967U,  971U,  977U,  983U,  989U,  991U,  997U,  1003U, 1007U, 1009U, 1013U, 1019U, 1021U, 1027U, 1031U, 1033U,
    1037U, 1039U, 1049U, 1051U, 1061U, 1063U, 1069U, 1073U, 1079U, 1081U, 1087U, 1091U, 1093U, 1097U, 1103U, 1109U, 1117U, 1121U, 1123U, 1129U, 1139U, 1147U, 1151U, 1153U,
    1157U, 1159U, 1163U, 1171U, 1181U, 1187U, 1189U, 1193U, 1201U, 1207U, 1213U, 1217U, 1219U, 1223U, 1229U, 1231U, 1237U, 1241U, 1247U, 1249U, 1259U, 1261U, 1271U, 1273U,
    1277U, 1279U, 1283U, 1289U, 1291U, 1297U, 1301U, 1303U, 1307U, 1313U, 1319U, 1321U, 1327U, 1333U, 1339U, 1343U, 1349U, 1357U, 1361U, 1363U, 1367U, 1369U, 1373U, 1381U,
    1387U, 1391U, 1399U, 1403U, 1409U, 1411U, 1417U, 1423U, 1427U, 1429U, 1433U, 1439U, 1447U, 1451U, 1453U, 1457U, 1459U, 1469U, 1471U, 1481U, 1483U, 1487U, 1489U, 1493U,
    1499U, 1501U, 1511U, 1513U, 1517U, 1523U, 1531U, 1537U, 1541U, 1543U, 1549U, 1553U, 1559U, 1567U, 1571U, 1577U, 1579U, 1583U, 1591U, 1597U, 1601U, 1607U, 1609U, 1613U,
    1619U, 1621U, 1627U, 1633U, 1637U, 1643U, 1649U, 1651U, 1657U, 1663U, 1667U, 1669U, 1679U, 1681U, 1691U, 1693U, 1697U, 1699U, 1703U, 1709U, 1711U, 1717U, 1721U, 1723U,
    1733U, 1739U, 1741U, 1747U, 1751U, 1753U, 1759U, 1763U, 1769U, 1777U, 1781U, 1783U, 1787U, 1789U, 1801U, 1807U, 1811U, 1817U, 1819U, 1823U, 1829U, 1831U, 1843U, 1847U,
    1849U, 1853U, 1861U, 1867U, 1871U, 1873U, 1877U, 1879U, 1889U, 1891U, 1901U, 1907U, 1909U, 1913U, 1919U, 1921U, 1927U, 1931U, 1933U, 1937U, 1943U, 1949U, 1951U, 1957U,
    1961U, 1963U, 1973U, 1979U, 1987U, 1993U, 1997U, 1999U, 2003U, 2011U, 2017U, 2021U, 2027U, 2029U, 2033U, 2039U, 2041U, 2047U, 2053U, 2059U, 2063U, 2069U, 2071U, 2077U,
    2081U, 2083U, 2087U, 2089U, 2099U, 2111U, 2113U, 2117U, 2119U, 2129U, 2131U, 2137U, 2141U, 2143U, 2147U, 2153U, 2159U, 2161U, 2171U, 2173U, 2179U, 2183U, 2197U, 2201U,
    2203U, 2207U, 2209U, 2213U, 2221U, 2227U, 2231U, 2237U, 2239U, 2243U, 2249U, 2251U, 2257U, 2263U, 2267U, 2269U, 2273U, 2279U, 2281U, 2287U, 2291U, 2293U, 2297U, 2309U};

// Make this NOT a multiple of 2, 3, 5, 7, or 11.
BigInteger forward11(const size_t &p) { return wheel11[p % 480U] + (p / 480U) * 2310U; }

size_t backward11(const BigInteger &n) { return std::distance(wheel11, std::lower_bound(wheel11, wheel11 + 480U, (size_t)(n % 2310U))) + 480U * (size_t)(n / 2310U) + 1U; }

inline BigInteger _forward2(const BigInteger &p) {
  return (p << 1U) | 1U;
}

inline BigInteger _backward2(const BigInteger &n) { return n >> 1U; }

inline BigInteger _forward3(const BigInteger &p) { return (p << 1U) + (~(~p | 1U)) - 1U; }

inline BigInteger _backward3(const BigInteger &n) { return ((~(~n | 1U)) / 3U) + 1U; }

BigInteger _forward5(const BigInteger &p) { return wheel5[(size_t)(p % 8U)] + (p / 8U) * 30U; }

BigInteger _backward5(const BigInteger &n) { return std::distance(wheel5, std::lower_bound(wheel5, wheel5 + 8U, (size_t)(n % 30U))) + 8U * (n / 30U) + 1U; }

BigInteger _forward7(const BigInteger &p) { return wheel7[(size_t)(p % 48U)] + (p / 48U) * 210U; }

BigInteger _backward7(const BigInteger &n) { return std::distance(wheel7, std::lower_bound(wheel7, wheel7 + 48U, n % 210U)) + 48U * (n / 210U) + 1U; }

BigInteger _forward11(const BigInteger &p) { return wheel11[(size_t)(p % 480U)] + (p / 480U) * 2310U; }

BigInteger _backward11(const BigInteger &n) { return std::distance(wheel11, std::lower_bound(wheel11, wheel11 + 480U, (size_t)(n % 2310U))) + 480U * (n / 2310U) + 1U; }

typedef BigInteger (*ForwardFn)(const BigInteger &);
inline ForwardFn forward(const Wheel &w) {
  switch (w) {
  case WHEEL2:
    return _forward2;
  case WHEEL3:
    return _forward3;
  case WHEEL5:
    return _forward5;
  case WHEEL7:
    return _forward7;
  case WHEEL11:
    return _forward11;
  case WHEEL1:
  default:
    return [](const BigInteger &n) -> BigInteger { return n; };
  }
}

inline ForwardFn backward(const Wheel &w) {
  switch (w) {
  case WHEEL2:
    return _backward2;
  case WHEEL3:
    return _backward3;
  case WHEEL5:
    return _backward5;
  case WHEEL7:
    return _backward7;
  case WHEEL11:
    return _backward11;
  case WHEEL1:
  default:
    return [](const BigInteger &n) -> BigInteger { return n; };
  }
}

inline size_t GetWheel5and7Increment(unsigned short &wheel5, unsigned long long &wheel7) {
  constexpr unsigned short wheel5Back = 1U << 9U;
  constexpr unsigned long long wheel7Back = 1ULL << 55U;
  unsigned wheelIncrement = 0U;
  bool is_wheel_multiple = false;
  do {
    is_wheel_multiple = (bool)(wheel5 & 1U);
    wheel5 >>= 1U;
    if (is_wheel_multiple) {
      wheel5 |= wheel5Back;
      ++wheelIncrement;
      continue;
    }

    is_wheel_multiple = (bool)(wheel7 & 1U);
    wheel7 >>= 1U;
    if (is_wheel_multiple) {
      wheel7 |= wheel7Back;
    }
    ++wheelIncrement;
  } while (is_wheel_multiple);

  return (size_t)wheelIncrement;
}

std::vector<BigInteger> SieveOfEratosthenes(const BigInteger &n) {
  std::vector<BigInteger> knownPrimes = {2U, 3U, 5U, 7U};
  if (n < 2U) {
    return std::vector<BigInteger>();
  }

  if (n < (knownPrimes.back() + 2U)) {
    const auto highestPrimeIt = std::upper_bound(knownPrimes.begin(), knownPrimes.end(), n);
    return std::vector<BigInteger>(knownPrimes.begin(), highestPrimeIt);
  }

  knownPrimes.reserve(std::expint(log((double)n)) - std::expint(log(2)));

  // We are excluding multiples of the first few
  // small primes from outset. For multiples of
  // 2, 3, and 5 this reduces complexity to 4/15.
  const size_t cardinality = backward5(n);

  // Create a boolean array "prime[0..cardinality]"
  // and initialize all entries it as true. Rather,
  // reverse the true/false meaning, so we can use
  // default initialization. A value in notPrime[i]
  // will finally be false only if i is a prime.
  std::unique_ptr<bool[]> uNotPrime(new bool[cardinality + 1U]());
  bool *notPrime = uNotPrime.get();

  // Get the remaining prime numbers.
  unsigned short wheel5 = 129U;
  unsigned long long wheel7 = 9009416540524545ULL;
  size_t o = 1U;
  for (;;) {
    o += GetWheel5and7Increment(wheel5, wheel7);

    const BigInteger p = forward3(o);
    if ((p * p) > n) {
      break;
    }

    if (notPrime[backward5(p)]) {
      continue;
    }

    knownPrimes.push_back(p);

    // We are skipping multiples of 2, 3, and 5
    // for space complexity, for 4/15 the bits.
    // More are skipped by the wheel for time.
    const BigInteger p2 = p << 1U;
    const BigInteger p4 = p << 2U;
    BigInteger i = p * p;

    // "p" already definitely not a multiple of 3.
    // Its remainder when divided by 3 can be 1 or 2.
    // If it is 2, we can do a "half iteration" of the
    // loop that would handle remainder of 1, and then
    // we can proceed with the 1 remainder loop.
    // This saves 2/3 of updates (or modulo).
    if ((p % 3U) == 2U) {
      notPrime[backward5(i)] = true;
      i += p2;
      if (i > n) {
        continue;
      }
    }

    for (;;) {
      if (i % 5U) {
        notPrime[backward5(i)] = true;
      }
      i += p4;
      if (i > n) {
        break;
      }

      if (i % 5U) {
        notPrime[backward5(i)] = true;
      }
      i += p2;
      if (i > n) {
        break;
      }
    }
  }

  for (;;) {
    const BigInteger p = forward3(o);
    if (p > n) {
      break;
    }

    o += GetWheel5and7Increment(wheel5, wheel7);

    if (notPrime[backward5(p)]) {
      continue;
    }

    knownPrimes.push_back(p);
  }

  return knownPrimes;
}

inline bool isMultiple(const BigInteger &p, const std::vector<uint16_t> &knownPrimes) {
  for (const uint16_t &prime : knownPrimes) {
    if (!(p % prime)) {
      return true;
    }
  }
  return false;
}

boost::dynamic_bitset<size_t> wheel_inc(std::vector<uint16_t> primes) {
  BigInteger radius = 1U;
  for (const uint16_t &i : primes) {
    radius *= i;
  }
  const uint16_t prime = primes.back();
  primes.pop_back();
  boost::dynamic_bitset<size_t> o;
  for (BigInteger i = 1U; i <= radius; ++i) {
    if (!isMultiple(i, primes)) {
      o.push_back(!(i % prime));
    }
  }
  o >>= 1U;

  return o;
}

std::vector<boost::dynamic_bitset<size_t>> wheel_gen(const std::vector<uint16_t> &primes) {
  std::vector<boost::dynamic_bitset<size_t>> output;
  std::vector<uint16_t> wheelPrimes;
  for (const uint16_t &p : primes) {
    wheelPrimes.push_back(p);
    output.push_back(wheel_inc(wheelPrimes));
  }
  return output;
}

inline size_t GetWheelIncrement(std::vector<boost::dynamic_bitset<size_t>> *inc_seqs) {
  size_t wheelIncrement = 0U;
  bool is_wheel_multiple = false;
  do {
    for (size_t i = 0U; i < inc_seqs->size(); ++i) {
      boost::dynamic_bitset<size_t> &wheel = (*inc_seqs)[i];
      is_wheel_multiple = wheel.test(0U);
      wheel >>= 1U;
      if (is_wheel_multiple) {
        wheel[wheel.size() - 1U] = true;
        break;
      }
    }
    ++wheelIncrement;
  } while (is_wheel_multiple);

  return wheelIncrement;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  WRITTEN BY ELARA (GPT) BELOW //
////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Utility to perform modular exponentiation
BigInteger modExp(BigInteger base, BigInteger exp, const BigInteger &mod) {
  BigInteger result = 1U;
  while (exp) {
    if (exp & 1U) {
      result = (result * base) % mod;
    }
    base = (base * base) % mod;
    exp >>= 1U;
  }

  return result;
}

// Perform Gaussian elimination on a binary matrix
void gaussianElimination(std::map<BigInteger, boost::dynamic_bitset<size_t>> *matrix) {
  size_t rows = matrix->size();
  size_t cols = matrix->begin()->second.size();
  std::vector<int> pivots(cols, -1);
  for (size_t col = 0U; col < cols; ++col) {
    auto colIt = matrix->begin();
    std::advance(colIt, col);

    auto rowIt = colIt;
    for (size_t row = col; row < rows; ++row) {
      if (rowIt->second[col]) {
        std::swap(colIt->second, rowIt->second);
        pivots[col] = row;
        break;
      }
      ++rowIt;
    }

    if (pivots[col] == -1) {
      continue;
    }

    const boost::dynamic_bitset<size_t> &c = colIt->second;
    rowIt = matrix->begin();
    for (size_t row = 0U; row < rows; ++row) {
      boost::dynamic_bitset<size_t> &r = rowIt->second;
      if ((row != col) && r[col]) {
        r ^= c;
      }
      ++rowIt;
    }
  }
}

// Compute the prime factorization modulo 2
boost::dynamic_bitset<size_t> factorizationVector(BigInteger num, const std::vector<uint16_t> &primes) {
  boost::dynamic_bitset<size_t> vec(primes.size(), false);
  for (size_t i = 0U; i < primes.size(); ++i) {
    bool count = false;
    const uint16_t &p = primes[i];
    while (!(num % p)) {
      num /= p;
      count = !count;
    }
    vec[i] = count;
    if (num == 1U) {
      break;
    }
  }
  if (num != 1U) {
    return boost::dynamic_bitset<size_t>();
  }

  return vec;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  WRITTEN BY ELARA (GPT) ABOVE //
////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Factorizer {
  std::mutex batchMutex;
  std::mutex smoothNumberMapMutex;
  std::default_random_engine rng;
  BigInteger toFactorSqr;
  BigInteger toFactor;
  BigInteger toFactorSqrt;
  BigInteger batchRange;
  BigInteger batchNumber;
  BigInteger batchBound;
  size_t wheelRatio;
  size_t primePartBound;
  bool isIncomplete;
  std::vector<uint16_t> primes;
  ForwardFn forwardFn;

  Factorizer(const BigInteger &tfsqr, const BigInteger &tf, const BigInteger &tfsqrt, const BigInteger &range, size_t nodeId, size_t wr, size_t ppb, const std::vector<uint16_t> &p,
             ForwardFn fn)
      : rng({}), toFactorSqr(tfsqr), toFactor(tf), toFactorSqrt(tfsqrt), batchRange(range), batchNumber(0U), batchBound((nodeId + 1U) * range), wheelRatio(wr), primePartBound(ppb),
        isIncomplete(true), primes(p), forwardFn(fn) {}

  BigInteger getNextBatch() {
    std::lock_guard<std::mutex> lock(batchMutex);

    if (batchNumber == batchRange) {
      isIncomplete = false;
      return batchBound;
    }

    return batchBound - (++batchNumber);
  }

  BigInteger getNextAltBatch() {
    std::lock_guard<std::mutex> lock(batchMutex);

    if (batchNumber == batchRange) {
      isIncomplete = false;
      return batchBound;
    }

    const BigInteger halfBatchNum = (batchNumber++ >> 1U);

    return batchBound - ((batchNumber & 1U) ? (BigInteger)(batchRange - halfBatchNum) : (BigInteger)(halfBatchNum + 1U));
  }

  BigInteger bruteForce(std::vector<boost::dynamic_bitset<size_t>> *inc_seqs) {
    // Up to wheel factorization, try all batches up to the square root of toFactor.
    for (BigInteger batchNum = getNextBatch(); isIncomplete; batchNum = getNextBatch()) {
      const BigInteger batchStart = batchNum * wheelRatio;
      const BigInteger batchEnd = (batchNum + 1U) * wheelRatio;
      for (BigInteger p = batchStart; p < batchEnd;) {
        p += GetWheelIncrement(inc_seqs);
        const BigInteger n = forwardFn(p);
        if (!(toFactor % n) && (n != 1U) && (n != toFactor)) {
          isIncomplete = false;
          return n;
        }
      }
    }

    return 1U;
  }

  BigInteger smoothCongruences(std::vector<boost::dynamic_bitset<size_t>> *inc_seqs, std::vector<BigInteger> *semiSmoothParts,
                               std::map<BigInteger, boost::dynamic_bitset<size_t>> *smoothNumberMap) {
    // Up to wheel factorization, try all batches up to the square root of toFactor.
    // Since the largest prime factors of these numbers is relatively small,
    // use the "exhaust" of brute force to produce smooth numbers for Quadratic Sieve.
    for (BigInteger batchNum = getNextAltBatch(); isIncomplete; batchNum = getNextAltBatch()) {
      const BigInteger batchStart = batchNum * wheelRatio;
      const BigInteger batchEnd = (batchNum + 1U) * wheelRatio;
      for (BigInteger p = batchStart; p < batchEnd;) {
        // Skip increments on the "wheels" (or "gears").
        p += GetWheelIncrement(inc_seqs);
        // Brute-force check if the sequential number is a factor.
        const BigInteger n = forwardFn(p);
        // If so, terminate this node and return the answer.
        if (!(toFactor % n) && (n != 1U) && (n != toFactor)) {
          isIncomplete = false;
          return n;
        }
        // Use the "exhaust" to produce smoother numbers.
        semiSmoothParts->push_back(n);
        // Batch this work, to reduce contention.
        if (semiSmoothParts->size() < primePartBound) {
          continue;
        }
        // Our "smooth parts" are smaller than the square root of toFactor.
        // We combine them semi-randomly, to produce numbers just larger than toFactor^(1/2).
        const BigInteger m = makeSmoothNumbers(semiSmoothParts, smoothNumberMap);
        // Check the factor returned.
        if (m != 1U) {
          // Gaussian elimination found a factor!
          isIncomplete = false;
          return m;
        }
      }
    }

    return 1U;
  }

  BigInteger makeSmoothNumbers(std::vector<BigInteger> *semiSmoothParts, std::map<BigInteger, boost::dynamic_bitset<size_t>> *smoothNumberMap) {
    // Factorize all "smooth parts."
    std::vector<BigInteger> smoothParts;
    std::map<BigInteger, boost::dynamic_bitset<size_t>> smoothPartsMap;
    for (const BigInteger &n : (*semiSmoothParts)) {
      const boost::dynamic_bitset<size_t> fv = factorizationVector(n, primes);
      if (fv.size()) {
        smoothPartsMap[n] = fv;
        smoothParts.push_back(n);
      }
    }
    // We can clear the thread's buffer vector.
    semiSmoothParts->clear();

    // This is the only nondeterminism in the algorithm.
    std::shuffle(smoothParts.begin(), smoothParts.end(), rng);

    // Now that smooth parts have been shuffled, just multiply down the list until they are larger than square root of toFactor.
    BigInteger smoothNumber = 1U;
    boost::dynamic_bitset<size_t> fv(primes.size(), false);
    for (size_t spi = 0U; spi < smoothParts.size(); ++spi) {
      const BigInteger &sp = smoothParts[spi];
      // This multiplies together the factorizations of the smooth parts
      // (producing the overall factorization of their multiplication)
      fv ^= smoothPartsMap[sp];
      smoothNumber *= sp;
      // Check if the number is big enough
      if (smoothNumber <= toFactorSqrt) {
        continue;
      }
      // For lock_guard scope
      if (true) {
        std::lock_guard<std::mutex> lock(smoothNumberMapMutex);
        auto it = smoothNumberMap->find(smoothNumber);
        if (it == smoothNumberMap->end()) {
          (*smoothNumberMap)[smoothNumber] = fv;
        }
      }
      // Reset "smoothNumber" and its factorization vector.
      smoothNumber = 1U;
      fv = boost::dynamic_bitset<size_t>(primes.size(), false);
    }
    // We're done with smoothParts.
    smoothParts.clear();

    // This entire next section is blocking (for Quadratic Sieve Gaussian elimination).
    std::lock_guard<std::mutex> lock(smoothNumberMapMutex);
    return findFactorViaGaussianElimination(toFactor, smoothNumberMap);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                   WRITTEN BY ELARA (GPT) BELOW                                                        //
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Find factor via Gaussian elimination
  BigInteger findFactorViaGaussianElimination(const BigInteger &target, std::map<BigInteger, boost::dynamic_bitset<size_t>> *smoothNumberMap) {
    // Perform Gaussian elimination
    gaussianElimination(smoothNumberMap);

    // Check for linear dependencies and find a congruence of squares
    std::vector<size_t> toStrike;
    auto iIt = smoothNumberMap->begin();
    for (size_t i = 0U; i < smoothNumberMap->size(); ++i) {
      boost::dynamic_bitset<size_t> &iRow = iIt->second;
      auto jIt = iIt;
      for (size_t j = i + 1U; j < smoothNumberMap->size(); ++j) {
        ++jIt;

        boost::dynamic_bitset<size_t> &jRow = jIt->second;
        if (iRow != jRow) {
          continue;
        }

        toStrike.push_back(j);

        // Compute x and y
        const BigInteger x = (iIt->first * jIt->first) % target;
        const BigInteger y = modExp(x, target / 2, target);

        // Check congruence of squares
        BigInteger factor = gcd(target, x + y);
        if ((factor != 1U) && (factor != target)) {
          return factor;
        }

        // Try x - y as well
        factor = gcd(target, x - y);
        if ((factor != 1U) && (factor != target)) {
          return factor;
        }
      }
      ++iIt;
    }

    // These numbers have been tried already:
    for (size_t i = 0U; i < toStrike.size(); ++i) {
      smoothNumberMap->erase(toStrike[i]);
    }

    return 1U; // No factor found
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                                                   WRITTEN BY ELARA (GPT) ABOVE                                                        //
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};

std::string find_a_factor(const std::string &toFactorStr, const bool &isConOfSqr, const size_t &nodeCount, const size_t &nodeId, size_t gearFactorizationLevel,
                          size_t wheelFactorizationLevel, double smoothnessBoundMultiplier) {
  // (At least) level 11 wheel factorization is baked into basic functions.
  if (!wheelFactorizationLevel) {
    wheelFactorizationLevel = 1U;
  } else if (wheelFactorizationLevel > 11U) {
    wheelFactorizationLevel = 11U;
    std::cout << "Warning: Wheel factorization limit is 11. (Parameter will be ignored and default to 11.)";
  }
  if (!gearFactorizationLevel) {
    gearFactorizationLevel = 1U;
  } else if (gearFactorizationLevel < wheelFactorizationLevel) {
    gearFactorizationLevel = wheelFactorizationLevel;
    std::cout << "Warning: Gear factorization level must be at least as high as wheel level. (Parameter will be ignored and default to wheel level.)";
  }

  // Convert from string.
  BigInteger toFactor(toFactorStr);

  // The largest possible discrete factor of "toFactor" is its square root (as with any integer).
  const BigInteger fullMaxBase = sqrt(toFactor);
  if (fullMaxBase * fullMaxBase == toFactor) {
    return boost::lexical_cast<std::string>(fullMaxBase);
  }

  // We only need to try trial division about as high as would be necessary for 4096 bits of semiprime.
  const BigInteger primeCeiling = (65536ULL < fullMaxBase) ? (BigInteger)65536ULL : fullMaxBase;
  BigInteger result = 1U;
  // This uses very little memory and time, to find primes.
  std::vector<BigInteger> bigPrimes = SieveOfEratosthenes(primeCeiling);
  // All of our primes are necessarily less than 16-bit
  std::vector<uint16_t> primes(bigPrimes.size());
  std::transform(bigPrimes.begin(), bigPrimes.end(), primes.begin(), [](const BigInteger &p) { return (uint16_t)p; });
  // "it" is the end-of-list iterator for a list up-to-and-including wheelFactorizationLevel.
  const auto itw = std::upper_bound(primes.begin(), primes.end(), wheelFactorizationLevel);
  const auto itg = std::upper_bound(primes.begin(), primes.end(), gearFactorizationLevel);
  const size_t wgDiff = std::distance(itw, itg);

  // This is simply trial division up to the ceiling.
  DispatchQueue dispatch(std::thread::hardware_concurrency());
  for (size_t primeIndex = 0U; (primeIndex < primes.size()) && (result == 1U); primeIndex += 64U) {
    dispatch.dispatch([&toFactor, &primes, &result, primeIndex]() {
      const size_t maxLcv = std::min(primeIndex + 64U, primes.size());
      for (size_t pi = primeIndex; pi < maxLcv; ++pi) {
        if (result != 1U) {
          return false;
        }
        const uint16_t &currentPrime = primes[pi];
        if (!(toFactor % currentPrime)) {
          result = currentPrime;
          return true;
        }
      }
      return false;
    });
  }
  dispatch.finish();
  // If we've checked all primes below the square root of toFactor, then it's prime.
  if ((result != 1U) || (toFactor <= (primeCeiling * primeCeiling))) {
    return boost::lexical_cast<std::string>(result);
  }

  // Set up wheel factorization (or "gear" factorization)
  std::vector<uint16_t> gearFactorizationPrimes(primes.begin(), itg);
  // Primes are only present in range above wheel factorization level
  primes = std::vector<uint16_t>(itg, primes.begin() + std::min(primes.size(), gearFactorizationPrimes.size() + (size_t)(smoothnessBoundMultiplier * log2(toFactor))));
  // From 1, this is a period for wheel factorization
  size_t biggestWheel = 1ULL;
  for (const uint16_t &wp : gearFactorizationPrimes) {
    biggestWheel *= (size_t)wp;
  }
  // These are "gears," for wheel factorization (with a "wheel" already in place up to 11).
  std::vector<boost::dynamic_bitset<size_t>> inc_seqs = wheel_gen(std::vector<uint16_t>(gearFactorizationPrimes.begin(), gearFactorizationPrimes.end()));
  // We're done with the lowest primes.
  const size_t MIN_RTD_LEVEL = gearFactorizationPrimes.size() - wgDiff;
  const Wheel SMALLEST_WHEEL = wheelByPrimeCardinal(MIN_RTD_LEVEL);
  // Skip multiples removed by wheel factorization.
  inc_seqs.erase(inc_seqs.begin(), inc_seqs.end() - wgDiff);
  gearFactorizationPrimes.clear();

  // Ratio of biggest vs. smallest wheel, for periodicity
  const size_t wheelRatio = biggestWheel / (size_t)SMALLEST_WHEEL;
  // Range per parallel node
  const BigInteger nodeRange = (((backward(SMALLEST_WHEEL)(fullMaxBase) + nodeCount - 1U) / nodeCount) + wheelRatio - 1U) / wheelRatio;
  // Same collection across all threads
  std::map<BigInteger, boost::dynamic_bitset<size_t>> smoothNumberMap;
  // This manages the work per thread
  Factorizer worker(toFactor * toFactor, toFactor, fullMaxBase, nodeRange, nodeId, wheelRatio, 1ULL << 14U, primes, forward(SMALLEST_WHEEL));

  const auto workerFn = [&toFactor, &inc_seqs, &isConOfSqr, &worker, &smoothNumberMap] {
    // inc_seq needs to be independent per thread.
    std::vector<boost::dynamic_bitset<size_t>> inc_seqs_clone;
    inc_seqs_clone.reserve(inc_seqs.size());
    for (const auto &b : inc_seqs) {
      inc_seqs_clone.emplace_back(b);
    }

    // "Brute force" includes extensive wheel multiplication and can be faster.
    if (!isConOfSqr) {
      return worker.bruteForce(&inc_seqs_clone);
    }

    // Different collection per thread;
    std::vector<BigInteger> semiSmoothParts;

    // While brute-forcing, use the "exhaust" to feed "smooth" number generation and check conguence of squares.
    return worker.smoothCongruences(&inc_seqs_clone, &semiSmoothParts, &smoothNumberMap);
  };

  const unsigned cpuCount = std::thread::hardware_concurrency();
  std::vector<std::future<BigInteger>> futures;
  futures.reserve(cpuCount);

  for (unsigned cpu = 0U; cpu < cpuCount; ++cpu) {
    futures.push_back(std::async(std::launch::async, workerFn));
  }

  for (unsigned cpu = 0U; cpu < cpuCount; ++cpu) {
    const BigInteger r = futures[cpu].get();
    if ((r > result) && (r != toFactor)) {
      result = r;
    }
  }

  return boost::lexical_cast<std::string>(result);
}
} // namespace Qimcifa

using namespace Qimcifa;

PYBIND11_MODULE(_find_a_factor, m) {
  m.doc() = "pybind11 plugin to find any factor of input";
  m.def("_find_a_factor", &find_a_factor, "Finds any nontrivial factor of input (or returns 1 if prime)");
}