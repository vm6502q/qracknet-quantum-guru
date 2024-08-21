# QrackNet

QrackNet API is open-source software to serve [unitaryfund/qrack](https://github.com/unitaryfund/qrack) gate-based quantum computer simulation "jobs" via a [PyQrack](https://github.com/unitaryfund/pyqrack)-like (lower-level) scripting language, as a Node.js-based web API. (Also see the [usage examples](https://github.com/vm6502q/qrack.net/blob/main/api/EXAMPLES.md), to help understand end-to-end workflows.

## QrackNet Script API Reference

QrackNet "script" takes the form of pure JSON "circuits" or "programs" that are each dispatched in an isolated simulation environment on the QrackNet API back end server.

### Script API Routes

###### `POST /api/qrack`

Accepts a script definition for the web API server to run, and returns a "Job ID." While certain method descriptions below highlight specific use cases for variable names, it is safe to assume that any numeric or boolean parameter (or parameter array entry) can be specified as a variable name from the output space.

- `program`: Array of method "instructions," executed in order from the first


###### `GET /api/qrack/:jobId`

Returns the status and "**output space**" of the job. All methods that return any output write it to the job (global) "output space," with names specified by the user that become schema for the "`output`" object.

### Glossary

**All glossary type precision is less than or equal to 53-bit, from JavaScript `Number` type.**

- `bitLenInt`: Bit-length integer - unsigned integer ID of qubit position in register.
- `bitCapInt`: Bit-capacity integer - unsigned integer permutation basis eigenstate value of a qubit register (typically "big integer," limited in input by JavaScript `Number` type).
- `real1`: Real number (1-dimensional) - floating-point real-valued number. **(JSON input precision: 32-bit IEEE floating-point.)**
- `Pauli`: Enum for Pauli bases - X is `1`, Z is `2`, Y is `3`, and "identity" is `0`.
- `quid`: Quantum (simulator) unique identifier - unsigned integer that indexes and IDs running simulators and neurons.

### Methods

Each method, as a line of the `program` array argument of the `POST /api/qrack` route has the following fields:

- `name`: String with the method name of the script instruction (as listed below, exact match, without markdown)
- `parameters`: Array of method parameters **in same order as defined in this reference document**. (In other words, supply "parameters" as **according to position in list**, not by "property" name, according this reference document, below.) `quid` arguments are always specified as variable names from the job's live "output space" of global variables, not number "literals." For "semi-classical" boolean methods, like `cland` below, boolean parameters can be supplied as variable names from the output space.
- `output`: **This is only used and required if a method returns a value.** Gives a name to the output of this method in the job's "output space." (If the variable name already exists, it will be overwritten.)
- `program`: **This is only used and required if a method is a classical boolean control structure.** When an classical boolean variable control condition evaluates to `true`, this nested `program` property is immediately dispatched like a general top-level QrackNet script, as a conditional subroutine.

#### Simulator Initialization

###### `init_general(bitLenInt length) -> quid`

**Returns** a `quid` representing a newly-initialized simulator optimized for "BQP-complete" (general) problems.

**As of the current pre-release version, prefer `init_qbdd` for general problems.** (`init_general` will ultimately serve this role, but its status is effectively experimental.)

- `length`: Number of qubits.


###### `init_stabilizer(bitLenInt length) -> quid`

**Returns** a `quid` representing a newly-initialized simulator optimized for ("hybrid") stabilizer problems (with recourse to universal circuit logic as a fallback).

- `length`: Number of qubits.


###### `init_qbdd(bitLenInt length) -> quid`

**Returns** a `quid` representing a newly-initialized simulator for low-entanglement problems (with "quantum binary decision diagrams" or "QBDD" simulation)

- `length`: Number of qubits.


###### `init_clone(quid sid) -> quid`

**Returns** a `quid` representing a newly-initialized clone of an existing simulator.

- `sid`: Simulator instance ID.


###### `destroy(quid sid)`

Destroys or releases a simulator instance.

- `sid`: Simulator instance ID.


###### `set_permutation(quid sid, bitCapInt p)`

Sets a simulator instance to the specified bit string permutation eigenstate (in measurement basis)

- `sid`: Simulator instance ID.
- `p`: Bit string permutation.


#### Random Number Generation

###### `seed(quid sid, unsigned s)`

Seeds the random number generator.

- `sid`: Simulator instance ID.
- `s`: Seed value.


#### Qubit Management

###### `allocate_qubit(quid sid, bitLenbitLenInt qid)`

Allocates a new qubit with a specific ID.

- `sid`: Simulator instance ID.
- `qid`: Qubit ID.


###### `release_qubit(quid sid, bitLenInt q) -> bool`

**Returns** true if the qubit ID is 'zeroed', and releases the qubit in any case.

- `sid`: Simulator instance ID.
- `qid`: Qubit ID.


###### `num_qubits(quid sid) -> bitLenInt`

**Returns** the total count of qubits in a simulator instance.

- `sid`: Simulator instance ID.


#### Measurement and Expectation Value Methods

###### `prob(quid sid, bitLenInt q) -> real1`

**Returns** The probability (from `0.0` to `1.0`) of the qubit being in the |1> state.

- `sid`: Simulator instance ID.
- `q`: Qubit ID.


###### `prob_rdm(quid sid, bitLenInt q) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) for probability of the qubit being in the |1> state, based on the "reduced density matrix" (with less overhead to calculate, for being "RDM").

- `sid`: Simulator instance ID.
- `q`: Qubit ID.


###### `perm_prob(quid sid, std::vector<bitLenInt> q, std::vector<Pauli> b) -> real1`

**Returns** the probability (upon measurement, in the corresponding joint Pauli basis) of collapsing into a specified permutation of a group of qubits.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of Pauli axes (for each qubit ID in `q`).


###### `perm_prob_rdm(quid sid, std::vector<bitLenInt> q, std::vector<Pauli> b, bool r) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) as to the probability (upon measurement, in the corresponding joint Pauli basis) of collapsing into a specified permutation of a group of qubits, based on the "reduced density matrix" (with less overhead to calculate, for being "RDM").

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of Pauli axes (for each qubit ID in `q`).
- `r`: "Rounding" option on/off, for `true`/`false`.


###### `fact_exp(quid sid, std::vector<bitLenInt> q, std::vector<long> s) -> real1`

**Returns** an expectation value by summing respective integers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).


###### `fact_exp_rdm(quid sid, std::vector<bitLenInt> q, std::vector<long> s, bool r) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) expectation value based on the "reduced density matrix," for an expectation value resulting from summing respective integers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array (with less overhead to calculate, for being "RDM").

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).
- `r`: "Rounding" option on/off, for `true`/`false`.


###### `fact_exp_fp(quid sid, std::vector<bitLenInt> q, std::vector<real1> s) -> real1`

**Returns** an expectation value by summing respective floating-point numbers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).


###### `fact_exp_fp_rdm(quid sid, std::vector<bitLenInt> q, std::vector<real1> s, bool r) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) expectation value based on the "reduced density matrix," for an expectation value resulting from summing respective floating-point numbers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array (with less overhead to calculate, for being "RDM").

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).
- `r`: "Rounding" option on/off, for `true`/`false`.


#### `unitary_exp(quid sid, std::vector<bitLenInt> q, std::vector<real1> b) -> real1`

**Returns** the single-qubit (3-parameter) operator expectation value for the array of qubits and (3-parameter unitary) bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of floating-point parameters representing `u` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with three times as many elements as `q`).


#### `matrix_exp(quid sid, std::vector<bitLenInt> q, std::vector<real1> b) -> real1`

**Returns** the single-qubit (2x2) operator expectation value for the array of qubits and (2x2 unitary matrix) bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: (Flat) array of floating-point values representing `mtrx` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with eight times as many elements as `q`).


#### `unitary_exp_ev(quid sid, std::vector<bitLenInt> q, std::vector<real1> b, std::vector<real1> e) -> real1`

**Returns** the single-qubit (3-parameter) operator expectation value for the array of qubits, (3-parameter unitary) bases, and (pairs of) expectation values.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of floating-point parameters representing `u` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with three times as many elements as `q`).
- `e`: Array of floating-point eigenvalues (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`)


#### `matrix_exp_ev(quid sid, std::vector<bitLenInt> q, std::vector<real1> b, std::vector<real1> e) -> real1`

**Returns** the single-qubit (2x2) operator expectation value for the array of qubits and (2x2 unitary matrix) bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: (Flat) array of floating-point values representing `mtrx` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with eight times as many elements as `q`).
- `e`: Array of floating-point eigenvalues (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`)


#### `pauli_exp(quid sid, std::vector<bitLenInt> q, std::vector<Pauli> b) -> real1`

**Returns** Pauli operator expectation value for the array of qubits and bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of Pauli axes (for each qubit ID in `q`).


###### `var(quid sid, std::vector<bitLenInt> q) -> real1`

**Returns** the variance associated to |0> and |1> as 1 and -1 expectation values of each qubit in "`q`."

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.


###### `var_rdm(quid sid, std::vector<bitLenInt> q) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) of the variance associated to |0> and |1> as 1 and -1 expectation values of each qubit in "`q`."

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `r`: "Rounding" option on/off, for `true`/`false`.


###### `fact_var(quid sid, std::vector<bitLenInt> q, std::vector<long> s) -> real1`

**Returns** the variance of the expectation value found by summing respective integers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).


###### `fact_var_rdm(quid sid, std::vector<bitLenInt> q, std::vector<long> s, bool r) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) variance based on the "reduced density matrix," for an expectation value resulting from summing respective integers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array (with less overhead to calculate, for being "RDM").

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).
- `r`: "Rounding" option on/off, for `true`/`false`.


###### `fact_var_fp(quid sid, std::vector<bitLenInt> q, std::vector<real1> s) -> real1`

**Returns** the variance of an expectation value found by summing respective floating-point numbers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).


###### `fact_var_fp_rdm(quid sid, std::vector<bitLenInt> q, std::vector<real1> s, bool r) -> real1`

**Returns** a "best-guess" (for near-Clifford simulation) variance based on the "reduced density matrix," for an expectation value resulting from summing respective floating-point numbers "`s`", associated to |0> and |1> respective states of each qubit in "`q`", across qubit basis ray permutations by probability "weight," with `s` strided in |0>/|1> pairs, as a flat array (with less overhead to calculate, for being "RDM").

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `s`: Array of integers (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`).
- `r`: "Rounding" option on/off, for `true`/`false`.

#### `unitary_var(quid sid, std::vector<bitLenInt> q, std::vector<real1> b) -> real1`

**Returns** the single-qubit (3-parameter) operator variance for the array of qubits and (3-parameter unitary) bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of floating-point parameters representing `u` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with three times as many elements as `q`).


#### `matrix_var(quid sid, std::vector<bitLenInt> q, std::vector<real1> b) -> real1`

**Returns** the single-qubit (2x2) variance for the array of qubits and (2x2 unitary matrix) bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: (Flat) array of floating-point values representing `mtrx` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with eight times as many elements as `q`).


#### `unitary_var_ev(quid sid, std::vector<bitLenInt> q, std::vector<real1> b, std::vector<real1> e) -> real1`

**Returns** the single-qubit (3-parameter) operator variance for the array of qubits, (3-parameter unitary) bases, and (pairs of) expectation values.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of floating-point parameters representing `u` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with three times as many elements as `q`).
- `e`: Array of floating-point eigenvalues (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`)


#### `matrix_var_ev(quid sid, std::vector<bitLenInt> q, std::vector<real1> b, std::vector<real1> e) -> real1`

**Returns** the single-qubit (2x2) operator variance for the array of qubits and (2x2 unitary matrix) bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: (Flat) array of floating-point values representing `mtrx` operations _from_ the desired basis _to_ the current basis (associated with each qubit ID in `q`, with eight times as many elements as `q`).
- `e`: Array of floating-point eigenvalues (associated to respective |0> and |1> states of each qubit ID in `q`, with twice as many elements as `q`)


#### `pauli_var(quid sid, std::vector<bitLenInt> q, std::vector<Pauli> b) -> real1`

**Returns** Pauli operator variance for the array of qubits and bases.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of Pauli axes (for each qubit ID in `q`).


###### `measure(quid sid, bitLenInt q) -> bool`

**Returns** a boolean (true for |1> and false for |0>) single-qubit measurement result, simulated according to the Born rules, collapsing the state.

- `sid`: Simulator instance ID.
- `q`: Qubit ID.


###### `force_measure(quid sid, bitLenInt q, bool r)`

Forces the measurement result of a single qubit (and so does not save it to the output space, from input). This is a pseudo-quantum operation. (However, quantum computers could similarly apply "post-selective" measurements, at exponential disadvantage compared to classical simulators.)

- `sid`: Simulator instance ID.
- `q`: Qubit ID.
- `r`: Desired measurement result to force.


###### `measure_basis(quid sid, std::vector<bitLenInt> q, std::vector<Pauli> b) -> bool`

**Returns** a single boolean measurement result upon collapse of an ensemble of qubits, jointly, via measurement, each in its specified Pauli basis.

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `b`: Array of Pauli axes (for each qubit ID in `q`).


###### `measure_all`(quid sid) -> bitCapInt`

**Returns** the bit string resulting from measuring all qubits according to the Born rules, collapsing the simulator state.

- `sid`: Simulator instance ID.


###### `measure_shots(quid sid, std::vector<bitLenInt> q, unsigned s) -> std::vector<bitCapInt>`

**Returns** an array of bit strings resulting from repeatedly measuring a set of qubits for a specified number of shots in the Z-basis, without collapsing the simulator state.

- `sid`: Simulator instance ID.
- `q`: Vector of qubit identifiers.
- `s`: Number of measurement shots.

###### `reset_all(quid sid)`

Resets the simulator state to the |0> permutation state for all qubits.

- `sid`: Simulator instance ID.


#### Single-qubit gates

##### Discrete single-qubit gates

**Each gate below takes the same two parameters:**

- `sid`: Simulator instance ID.
- `q`: Qubit ID.

These are the gates:

- `x(quid sid, bitLenInt q)`
- `y(quid sid, bitLenInt q)`
- `z(quid sid, bitLenInt q)`
- `h(quid sid, bitLenInt q)`
- `s(quid sid, bitLenInt q)`
- `sx(quid sid, bitLenInt q)`
- `sy(quid sid, bitLenInt q)`
- `t(quid sid, bitLenInt q)`
- `adjs(quid sid, bitLenInt q)`
- `adjsx(quid sid, bitLenInt q)`
- `adjsy(quid sid, bitLenInt q)`
- `adjt(quid sid, bitLenInt q)`

##### Parameterized single-qubit gates

###### `u(quid sid, bitLenInt q, real1 theta, real1 phi, real1 lambda)`

General 3-parameter unitary single-qubit gate (covers all possible single-qubit gates)

- `sid`: Simulator instance ID.
- `q`: Qubit ID.
- `theta`: angle (radians)
- `phi`: angle (radians)
- `lambda`: angle (radians)


###### `mtrx(quid sid, std::vector<real1> m, bitLenInt q)`

General 2x2 unitary matrix operator single-qubit gate (covers all possible single-qubit gates)

- `sid`: Simulator instance ID.
- `m`: 8 floating-point numbers in a "flat" array representating alternating real/imaginary components of a (row-major) 2x2 complex unitary matrix.
- `q`: Qubit ID.


###### `r(quid sid, double phi, bitLenInt q, Pauli b)`

Rotates qubit by the specified angle around the specified Pauli axis.

- `sid`: Simulator instance ID.
- `phi`: Angle of rotation in radians.
- `q`: Qubit ID.
- `b`: Pauli axis of rotation.


#### Multi-controlled single-qubit gates

**"MC"** gates are activated by **|1>** control qubit states.
**"MAC"** gates are activated by **|0>** control qubit states.

##### Discrete multi-controlled single-qubit gates

**Each gate below takes the same three parameters:**

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Qubit ID.

These are the gates:

- `mcx(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mcy(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mcz(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mch(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mcs(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mct(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mcadjs(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mcadjt(quid sid, std::vector<bitLenInt> c, bitLenInt q)`

- `macx(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `macy(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `macz(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mach(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `macs(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `mact(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `macadjs(quid sid, std::vector<bitLenInt> c, bitLenInt q)`
- `macadjt(quid sid, std::vector<bitLenInt> c, bitLenInt q)`

##### Parameterized mult-controlled single-qubit gates

###### `mcu(quid sid, std::vector<bitLenInt> c, bitLenInt q, real1 theta, real1 phi, real1 lambda)`
###### `macu(quid sid, std::vector<bitLenInt> c, bitLenInt q, real1 theta, real1 phi, real1 lambda)`

General 3-parameter unitary single-qubit target with arbitrary number of control qubits (covers all possible single-qubit target "payloads")

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Qubit ID.
- `theta`: angle
- `phi`: angle
- `lambda`: angle


###### `mcmtrx(quid sid, std::vector<bitLenInt> c, std::vector<real1> m, bitLenInt q)`
###### `macmtrx(quid sid, std::vector<bitLenInt> c, std::vector<real1> m, bitLenInt q)`

General 2x2 unitary matrix operator single-qubit target with arbitrary number of control qubits (covers all possible single-qubit target "payloads")

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `m`: 8 floating-point numbers representating alternating real/imaginary components of a (row-major) 2x2 complex unitary matrix.
- `q`: Qubit ID.


###### `mcr(quid sid, std::vector<bitLenInt> c, double phi, bitLenInt q, Pauli b)`

Rotates qubit by the specified angle around the specified Pauli axis.

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `phi`: Angle of rotation in radians.
- `q`: Qubit ID.
- `b`: Pauli axis of rotation.


##### Special multi-controlled single-qubit gates

###### `ucmtrx(quid sid, std::vector<bitLenInt> c, std::vector<real1> m, bitLenInt q, bitLenInt p)`

Multi-controlled gate that activates only for the specified permutation of controls

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `m`: 8 floating-point numbers in a "flat" array representating alternating real/imaginary components of a (row-major) 2x2 complex unitary matrix.
- `q`: Qubit ID.


###### `multiplex_1qb_mtrx(quid sid, std::vector<bitLenInt> c, std::vector<real1> m, bitLenInt q)`

Multi-controlled, single-target multiplexer gate

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `m`: For **each** permutation of control qubits, numbered from 0 as ascending binary (unsigned) integers, in an overall "flat" array, 8 floating-point numbers representating alternating real/imaginary components of a (row-major) 2x2 complex unitary matrix.
- `q`: Qubit ID.


#### Coalesced single-qubit gates

**These optimized gates apply the same Pauli operator to all specified qubits and take the same two arguments. This is optimized, compared to separate Pauli gates.**

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.

These are the gates:

- `mx(quid sid, std::vector<bitLenInt> q)`
- `my(quid sid, std::vector<bitLenInt> q)`
- `mz(quid sid, std::vector<bitLenInt> q)`


#### Multi-qubit Pauli exponentiation gates

###### `exp(quid sid, std::vector<bitLenInt> q, std::vector<Pauli> b, real1 phi)`

Applies e^{-i * theta * b}, exponentiation of the specified Pauli operator corresponding to each qubit

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.
- `phi`: Angle of the rotation in radians.


###### `mcexp(quid sid, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<Pauli> b, real1 phi)`

Applies e^{-i * theta * b}, exponentiation of the specified Pauli operator corresponding to each qubit

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of qubit IDs.
- `phi`: Angle of the rotation in radians.

#### Swap gate variants

###### `swap(quid sid, bitLenInt q1, bitLenInt q2)`

Swap the two input qubits

- `sid`: Simulator instance ID.
- `q1`: Qubit ID (1).
- `q2`: Qubit ID (2).


###### `iswap(quid sid, bitLenInt q1, bitLenInt q2)`

Swap the two input qubits and apply a factor of "i" if their states differ

- `sid`: Simulator instance ID.
- `q1`: Qubit ID (1).
- `q2`: Qubit ID (2).


###### `adjiswap(quid sid, bitLenInt q1, bitLenInt q2)`

Swap the two input qubits and apply a factor of "-i" if their states differ (inverse of `iswap`)

- `sid`: Simulator instance ID.
- `q1`: Qubit ID (1).
- `q2`: Qubit ID (2).


###### `fsim(quid sid, real1 theta, real1 phi bitLenInt q1, bitLenInt q2)`

Apply "fsim," which is a phased swap-like gate that is useful in fermionic simulation

- `sid`: Simulator instance ID.
- `theta`: angle (radians) (1)
- `phi`: angle (radians) (2)
- `q1`: Qubit ID (1).
- `q2`: Qubit ID (2).


###### `mcswap(quid sid, std::vector<bitLenInt> c, bitLenInt q1, bitLenInt q2)`

If controls are all |1>, swap the two input qubits

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q1`: Qubit ID (1).
- `q2`: Qubit ID (2).


###### `macswap(quid sid, std::vector<bitLenInt> c, bitLenInt q1, bitLenInt q2)`

If controls are all |0>, swap the two input qubits

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q1`: Qubit ID (1).
- `q2`: Qubit ID (2).


#### Boolean (Toffoli) gates

**Each gate below takes the same four parameters:**

- `sid`: Simulator instance ID.
- `qi1`: Input qubit ID (1).
- `qi2`: Input qubit ID (2).
- `qo`: Output qubit ID.

These are the gates:

- `and(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)`
- `or(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)`
- `xor(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)`
- `nand(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)`
- `nor(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)`
- `xnor(quid sid, bitLenInt qi1, bitLenInt qi2, bitLenInt qo)`

#### Boolean (Semi-Classical) gates

**Each gate below takes the same four parameters:**

- `sid`: Simulator instance ID.
- `ci`: Input classical bit value (literal value or boolean output space variable name).
- `qi`: Input qubit ID.
- `qo`: Output qubit ID.

These are the gates:

- `cland(quid sid, bool ci, bitLenInt qi, bitLenInt qo)`
- `clor(quid sid, bool ci, bitLenInt qi, bitLenInt qo)`
- `clxor(quid sid, bool ci, bitLenInt qi, bitLenInt qo)`
- `clnand(quid sid, bool ci, bitLenInt qi, bitLenInt qo)`
- `clnor(quid sid, bool ci, bitLenInt qi, bitLenInt qo)`
- `clxnor(quid sid, bool ci, bitLenInt qi, bitLenInt qo)`

**Boolean parameters in the methods above can be specified as either values or output space variable names.**

There is a special method for saving `true` or `false` to an ouput method variable:

###### `write_bool(bool b) -> bool`

Write a boolean value to the output space, with a variable name

- `b`: Bool to write to `output` variable name.


#### Quantum Fourier transform

###### `qft(quid sid, std::vector<bitLenInt> q)`

Acts the quantum Fourier transform on the specified set of qubits (without terminal swap gates to reverse bit order)

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.


###### `iqft(quid sid, std::vector<bitLenInt> q)`

Acts the inverse of the quantum Fourier transform on the specified set of qubits (without terminal swap gates to reverse bit order)

- `sid`: Simulator instance ID.
- `q`: Array of qubit IDs.

#### Arithmetic Logic Unit (ALU)

###### `add(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

Add classical integer to quantum integer (in-place)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `a`: Classical integer operand.


###### `sub(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

Subtract classical integer from quantum integer (in-place)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `a`: Classical integer operand.


###### `adds(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

Add classical integer to quantum integer (in-place) and set an overflow flag

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `a`: Classical integer operand.
- `s`: Qubit ID of overflow flag.


###### `subs(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

Subtract classical integer from quantum integer (in-place) and set an overflow flag

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `a`: Classical integer operand.
- `s`: Qubit ID of overflow flag.


###### `mcadd(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

If controls are all |1>, add classical integer to quantum integer (in-place)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `a`: Classical integer operand.


###### `mcsub(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

If controls are all |1>, subtract classical integer from quantum integer (in-place)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `a`: Classical integer operand.


###### `mul(quid sid, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a)`

Multiply quantum integer by classical integer (in-place)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `o`: Array of overflow qubit IDs.
- `a`: Classical integer operand.


###### `div(quid sid, std::vector<bitLenInt> q, bitCapInt a)`

Divide quantum integer by classical integer (in-place)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `o`: Array of overflow qubit IDs.
- `a`: Classical integer operand.


###### `muln(quid sid, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a, bitCapInt m)`

Multiply quantum integer by classical integer (out-of-place, with modulus)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `o`: Array of output qubit IDs.
- `a`: Classical integer operand.
- `m`: Modulo base.


###### `divn(quid sid, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a, bitCapInt m)`

Divide quantum integer by classical integer (out-of-place, with modulus)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `o`: Array of output qubit IDs.
- `a`: Classical integer operand.
- `m`: Modulo base.


###### `pown(quid sid, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a, bitCapInt m)`

Raise a classical base to a quantum power (out-of-place, with modulus)

- `sid`: Simulator instance ID.
- `q`: Array of target qubit IDs.
- `o`: Array of output qubit IDs.
- `a`: Classical integer operand.
- `m`: Modulo base.


###### `mcmul(quid sid, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a)`

If controls are all |1>, multiply quantum integer by classical integer (in-place)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `o`: Array of overflow qubit IDs.
- `a`: Classical integer operand.


###### `mcdiv(quid sid, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a)`

If controls are all |1>, divide quantum integer by classical integer (in-place)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `o`: Array of overflow qubit IDs.
- `a`: Classical integer operand.


###### `mcmuln(quid sid, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a, bitCapInt m)`

If controls are all |1>, multiply quantum integer by classical integer (out-of-place, with modulus)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `o`: Array of output qubit IDs.
- `a`: Classical integer operand.
- `m`: Modulo base.


###### `mcmuln(quid sid, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a, bitCapInt m)`

If controls are all |1>, divide quantum integer by classical integer (out-of-place, with modulus)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `o`: Array of output qubit IDs.
- `a`: Classical integer operand.
- `m`: Modulo base.


###### `mcpown(quid sid, std::vector<bitLenInt> c, std::vector<bitLenInt> q, std::vector<bitLenInt> o, bitCapInt a, bitCapInt m)`

If controls are all |1>, raise a classical base to a quantum power (out-of-place, with modulus)

- `sid`: Simulator instance ID.
- `c`: Array of control qubit IDs.
- `q`: Array of target qubit IDs.
- `o`: Array of output qubit IDs.
- `a`: Classical integer operand.
- `m`: Modulo base.


### Quantum Neuron Activation Functions

Quantum neurons can use different activation functions, as defined in the QNeuronActivationFn enumeration:

- `Sigmoid (Default)`: Standard sigmoid activation function.
- `ReLU`: Rectified Linear Unit activation function.
- `GeLU`: Gaussian Error Linear Unit activation function.
- `Generalized Logistic`: A variation of the sigmoid function with tunable sharpness.
- `Leaky ReLU`: Leaky version of the Rectified Linear Unit activation function.

### Quantum Neuron Methods

#### Initialization

###### `init_qneuron(quid sid, std::vector<bitLenInt> c, bitLenInt q, QNeuronActivationFn f, real1 a, real1 tol) -> quid`

Initializes a quantum neuron with specified parameters.

- `sid`: Simulator instance ID.
- `c`: List of control qubits for input.
- `q`: Target qubit for output.
- `f`: Activation function.
- `a`: Alpha parameter (specific to certain activation functions).
- `tol`: Tolerance for neuron activation.

#### Cloning

###### `clone_qneuron(quid nid) -> quid`

Clones an existing quantum neuron.

- `nid`: Neuron instance ID.

#### Destruction

###### `destroy_qneuron(quid nid)`

Destroys a quantum neuron.

- `nid`: Neuron instance ID.

#### Configuration

###### `set_qneuron_angles(quid nid, std::vector<real1> angles)`

Sets the RY-rotation angle parameters for the quantum neuron.

- `nid`: Neuron instance ID.
- `angles`: Vector of angles for each input permutation.


###### `get_qneuron_angles(quid nid) -> std::vector<real1>`

Retrieves the RY-rotation angle parameters of the quantum neuron.

- `nid`: Neuron instance ID.
- `Returns`: Vector of angles.


###### `set_qneuron_alpha(quid nid, real1 alpha)`

Sets the leakage parameter for leaky quantum neuron activation functions.

- `nid`: Neuron instance ID.
- `alpha`: Leakage parameter value.


###### `set_qneuron_activation_fn(quid nid, QNeuronActivationFn f)`

Sets the activation function of a quantum neuron.

- `nid`: Neuron instance ID.
- `f`: Activation function.

#### Learning and Inference

###### `qneuron_predict(quid nid, bool e, bool r) -> real1`

**Returns** an inference result using the quantum neuron.

- `nid`: Neuron instance ID.
- `e`: Expected boolean inference result.
- `r`: Boolean to reset/leave the output qubit state before inference


###### `qneuron_unpredict(quid nid, bool e) -> real1`

**Returns** an inference result using the inverse operation of neuron inference.

- `nid`: Neuron instance ID.
- `e`: Expected boolean inference result.


###### `qneuron_learn_cycle(quid nid, bool e) -> real1`

**Returns** an inference result using the quantum neuron, training for one epoch and uncomputing intermediate effects.

- `nid`: Neuron instance ID.
- `e`: Expected boolean inference result.


###### `qneuron_learn(quid nid, real1 eta, bool e, bool r)`

Trains the quantum neuron for one epoch.

- `nid`: Neuron instance ID.
- `eta`: Learning rate.
- `e`: Expected boolean inference result.
- `r`: Boolean to reset/keep the output qubit state before learning


###### `qneuron_learn_permutation(quid nid, real1 eta, bool e, bool r)`

Trains the quantum neuron for one epoch, assuming a Z-basis eigenstate input.

- `nid`: Neuron instance ID.
- `eta`: Learning rate.
- `e`: Expected boolean inference result.
- `r`: Boolean to reset/keep the output qubit state before learning

#### Schmidt Decomposition Rounding Parameter and Near-Clifford Rounding (Approximation)

Fidelity methods are strictly `double` precision, not explicitly `real1`.

###### `set_sdrp(quid sid, double sdrp)`

Set the "Schmidt decomposition rounding parameter" ("SDRP"). If "reactive separation" option is on (as by default) the parameter will be automatically applied in multi-qubit gate operations. (See [arXiv:2304.14969](https://arxiv.org/abs/2304.14969), by Strano and the Qrack and Unitary Fund teams, on comparative benchmarks relative to Qrack.)

- `sid`: Simulator instance ID.
- `sdrp`: Schmidt decomposition rounding parameter (0 to 1 range, defaults to real1 "epsilon")


###### `set_ncrp(quid sid, double ncrp)`

Set the "near-Clifford rounding parameter" ("NCRP"). When near-Clifford gate set is being used (such as general Clifford gates plus general single-qubit phase gates), this value controls how "severe" any non-Clifford effect needs to be, to be taken into consideration, or else (internally managed and applied) non-Clifford gates might be ignored.

- `sid`: Simulator instance ID.
- `ncrp`: Near-Clifford rounding parameter (0 to 1 range, defaults to real1 "epsilon")


###### `get_unitary_fidelity(quid sid) -> double`

Report a close theoretical estimate of fidelity, as potentially reduced by "SDRP" (Credit to Andrea Mari for research at Unitary Fund, in [arXiv:2304.14969](https://arxiv.org/abs/2304.14969))

- `sid`: Simulator instance ID.
- `sdrp`: Schmidt decomposition rounding parameter (0 to 1 range, defaults to real1 "epsilon")


###### `reset_unitary_fidelity(quid sid)`

Reset the "SDRP" fidelity tracker to 1.0 fidelity ("ideal"), before continuing fidelity calculation. (Some wholly-destructive measurement and state preparation operations and side effects might automatically reset the fidelity tracker to 1.0, as well, though the attempt in design is to do so unobstructively to the utility of the fidelity tracking function in typical use.)

- `sid`: Simulator instance ID.


###### `set_reactive_separate(quid sid)`

Turn "reactive separation" optimization on/off with true/false (default: on/true). (Some subjectively "high-entanglement" circuits will run more quickly with "reactive separation" off.)

- `sid`: Simulator instance ID.


###### `set_t_injection(quid sid)`

Turn "near-Clifford" simulation techniques (for not just "`t`" gate, but "`r`" around Pauli Z axis in general) on/off, with true/false (default: on/true). (Near-clifford techniques are memory-efficient but might take very much longer execution time, without any "rounding" approximations applied, than other simulation techniques like state vector.)

- `sid`: Simulator instance ID.


#### Classical control

These methods modify or base control upon boolean variables in the output space. Note that if you need arithmetic operations on integer variables in the output space (like for loop control), **use an auxiliary (non-stabilizer) simulator instance with quantum ALU operations and measurement output,** since quantum ALU operations are handled in an entirely efficient manner when the input state is an eigenstate equivalent to a classical bit state.

###### `not(bool b)`

Applies an in-place "not" operation to a boolean variable named by `b` in the output space (so `true` becomes `false`, and `false` becomes `true`).

- `b`: Boolean variable name.


###### `cif(bool b)`

**Dispatches** the additional `program` property of the method object as a subroutine if the boolean variable in the output space named by `b` is `true`.

- `b`: Boolean variable name.


###### `for(bitCapInt i) -> bitCapInt`

**Iterates and returns** a loop control variable, starting at 0, completing `i` total count of loop iterations.
**Dispatches** the loop body once for each iteration.

- `i`: Integer literal or variable name.


## QrackNet Usage Examples

This guide assumes authenticated access to QrackNet, holding a necessary session cookie if required. (Sometimes a session cookie is not required.)

### Flipping a (Simulated) Quantum Coin

Simulate a quantum coin flip using a single qubit in superposition. Use the `POST /api/qrack` route with the script below. It initializes a one-qubit simulator, applies a Hadamard gate, and measures the qubit. The first operation creates a simulator instance (`qsim`), which is then used in subsequent operations. Qubits are indexed starting at 0.

```json
{
    "program": [
        { "name": "init_general", "parameters": [1], "output": "qsim" },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "measure", "parameters": ["qsim", 0], "output": "result" }
    ]
}
```

Upon submission, a response with a unique job ID is received. Later, retrieve the job's outcome (true or false) using `GET /api/qrack/{jobId}`.

### Flipping Two Entangled "Coins"

To demonstrate entanglement, we use a Bell pair. This script initializes a two-qubit simulator, applies a Hadamard gate to the first qubit, entangles them using a CNOT gate, and measures the entangled state using multiple shots.

```json
{
    "program": [
        { "name": "init_general", "parameters": [2], "output": "qsim" },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "mcx", "parameters": ["qsim", [0], 1] },
        { "name": "measure_shots", "parameters": ["qsim", [0, 1], 8], "output": "result" }
    ]
}
```

Submit the script and use the returned job ID to check the results. The outcomes should demonstrate correlated states of the qubits, reflecting entanglement.

### Quantum teleportation

It is particularly instructive to see a script that can simulate the "quantum teleportation" algorithm, in QrackNet syntax:
```json
{
    "program" : [
        { "name": "init_qbdd", "parameters": [3], "output": "qsim" },
        { "name": "h", "parameters": ["qsim", 1] },
        { "name": "mcx", "parameters": ["qsim", [1], 2] },
        { "name": "u", "parameters": ["qsim", 0, 0.3, 2.2, 1.4] },
        { "name": "prob", "parameters": ["qsim", 0], "output": "aliceZ" },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "prob", "parameters": ["qsim", 0], "output": "aliceX" },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "mcx", "parameters": ["qsim", [0], 1] },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "measure", "parameters": ["qsim", 0], "output": "aliceM0" },
        { "name": "measure", "parameters": ["qsim", 1], "output": "aliceM1" },
        { "name": "cif", "parameters": ["aliceM0"], "program": { "name": "z", "parameters": ["qsim", 2] }},
        { "name": "cif", "parameters": ["aliceM1"], "program": { "name": "x", "parameters": ["qsim", 2] }},
        { "name": "prob", "parameters": ["qsim", 2], "output": "bobZ" },
        { "name": "h", "parameters": ["qsim", 2] },
        { "name": "prob", "parameters": ["qsim", 2], "output": "bobX" }
    ]
}
```
"Alice" and "Bob" each have different qubits in a Bell pair; Alice entangles a (1-qubit) "message" to send to Bob (characterized by `"aliceZ"` and `"aliceX"`); Alice measures her two qubits and "sends" the classical bit results to Bob; Bob acts up to two classically-controlled operations on his remaining Bell pair qubit, to "receive" the message (in `"bobZ"` and `"bobX"`). Refer to any good textbook or [encyclopedia](https://en.wikipedia.org/wiki/Quantum_teleportation) to further "unpack" and explain the program operations used above, though we provide it as an authoritative reference implementation in QrackNet script specifically, of the common thought experiment where "Alice" transmits a message to "Bob" with a Bell pair.

`"aliceZ"` and `"aliceX"` are the state of the original "message" qubit to teleport, and they should match the `"bobZ"` and `"bobX"` message received by the end of the algorithm and program in the job results:
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "qsim": 0,
            "aliceZ": 0.022331755608320236,
            "aliceX": 0.5869569778442383,
            "aliceM0": true,
            "aliceM1": true,
            "bobZ": 0.022331759333610535,
            "bobX": 0.5869571566581726
        }
    }
}
```

### Grover's search algorithm (AKA "amplitude amplification")

Encoding "oracles" to "solve" with Grover's search algorithm (or "amplitude amplification") is relatively easy with use of the QrackNet script's "arithmetic logic unit" ("ALU") methods. We initialize a simulator, then we prepare it in initial maximal superposition, then the third code section below is the "oracle" we are trying to "invert" (or "search") using the advantaged quantum algorithm. The oracle tags only the bit string "`3`" as a "match." Then, according to the general amplitude amplification algorithm, we "uncompute" and reverse phase on the original input state, re-prepare with `h` gates, and iterate for `floor(sqrt(pow(2, n)) * PI / 4)` total repetitions to search an oracle of `n` qubits width with a single accepted "match" to the oracle:

```json
{
    "program" : [
        { "name": "init_qbdd", "parameters": [3], "output": "qsim" },

        { "name": "for", "parameters": [3], "output": "i", "program": [{ "name": "h", "parameters": ["qsim", "i"] }]},

        { "name": "for", "parameters": [2], "program": [
            { "name": "sub", "parameters": ["qsim", [0, 1, 2], 3] },
            { "name": "x", "parameters": ["qsim", 2] },
            { "name": "macz", "parameters": ["qsim", [0, 1], 2] },
            { "name": "x", "parameters": ["qsim", 2] },
            { "name": "add", "parameters": ["qsim", [0, 1, 2], 3] },

            { "name": "for", "parameters": [3], "output": "i", "program": [{ "name": "h", "parameters": ["qsim", "i"] }]},
            { "name": "x", "parameters": ["qsim", 2] },
            { "name": "macz", "parameters": ["qsim", [0, 1], 2] },
            { "name": "x", "parameters": ["qsim", 2] },
            { "name": "for", "parameters": [3], "output": "i", "program": [{ "name": "h", "parameters": ["qsim", "i"] }]}
        ] },
 
        { "name": "measure_shots", "parameters": ["qsim", [0, 1, 2], 8], "output": "result" }
    ]
}
```

Surely enough, our job response will look like this (with a typically very small probabilistic element to the measurement "shot" output distribution):
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "i": 3,
            "qsim": 0,
            "result": [
                0,
                3,
                3,
                3,
                3,
                3,
                3,
                3
            ]
        }
    }
}
```

### GHZ-state preparation

The QrackNet server is likely to successfully run even 50-qubit or larger GHZ state preparation circuits (with measurement), though reporting precision for results might currently be limited to about 32 bits. A 50-qubit GHZ-state preparation definitely won't work with `init_general` (which will ultimately attempt to allocate ~50 qubits of state vector representation) but it's likely to work with `init_stabilizer` and `init_qbdd`, as both are optimized (differently) for this GHZ-state case.

We can think of the state preparation as an `h` gate followed by a loop over `mcx` gates with single control qubits, but we need to "manually unroll" the conceptual loop (for 4 qubits in total, for this example):
```json
{
    "program" : [
        { "name": "init_stabilizer", "parameters": [4], "output": "qsim" },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "mcx", "parameters": ["qsim", [0], 1] },
        { "name": "mcx", "parameters": ["qsim", [0], 2] },
        { "name": "mcx", "parameters": ["qsim", [0], 3] },
        { "name": "measure_shots", "parameters": ["qsim", [0, 1, 2, 3], 16], "output": "result" }
    ]
}
```
If we exceed Clifford gate set "vocabulary" in any single line of the script, we are likely (but not guaranteed) to fall back to state vector with `init_stabilizer`, which in a case like with 50 qubits will lead to a (self-recovering, handled) "server crash." On the other hand, `init_qbdd` can also efficiently handle this state preparation without the requirement to constrain one's gate set vocabulary to pure Clifford gates. (This might make QBDD the more powerful option in many cases where GHZ-state preparation is only one "subroutine" in a larger quantum algorithm with a universal gate set.)

This an example result, for the above job:
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "qsim": 0,
            "result": [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                15,
                15,
                15,
                15,
                15,
                15,
                15,
                15
            ]
        }
    }
}
```

### Quantum Fourier transform

The "quantum Fourier transform" ("QFT") is exposed as a complete subroutine via a single method call:
```json
{
    "program" : [
        { "name": "init_qbdd", "parameters": [3], "output": "qsim" },
        { "name": "qft", "parameters": ["qsim", [0, 1, 2]] },
        { "name": "measure_shots", "parameters": ["qsim", [0, 1, 2], 8], "output": "result" }
    ]
}
```
`iqft` instruction is similarly the "inverse QFT", with the same syntax. These method do **not** apply terminal or leading `swap` gates to recover the canonical element order of the "discrete Fourier transform" ("DFT"), as many applications of the subroutine do not require this step (which is commonly included in other descriptions or definitions of the algorithm). To recover DFT amplitude ordering, further reverse the order of qubits in the result, before or else after the `qft`, with `swap` gates. (`swap` gates are commonly implemented via qubit label swap, hence they are virtually computationally "free," with no loss of generality in the case of a "fully-connected" qubit topologies, like in all QrackNet simulators.)


### Shor's algorithm (quantum period finding subroutine)

The quantum period finding subroutine of Shor's algorithm is practically "trivial" to express and simulate with QrackNet script:
```json
{
    "program" : [
        { "name": "init_qbdd", "parameters": [8], "output": "qsim" },

        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "h", "parameters": ["qsim", 1] },
        { "name": "h", "parameters": ["qsim", 2] },
        { "name": "h", "parameters": ["qsim", 3] },

        { "name": "pown", "parameters": ["qsim", [0, 1, 2, 3], [4, 5, 6, 7], 6, 15] },
        { "name": "iqft", "parameters": ["qsim", [0, 1, 2, 3]] },

        { "name": "measure_shots", "parameters": ["qsim", [0, 1, 2, 3], 16], "output": "result" }
    ]
}
```
We start the subroutine by preparing a maximal superposition of the input register. The (out-of-place operation) quantum "power modulo 'n'" base is chosen at random so it does not have a greater common divisor higher than 1 with the number to factor (`15`, in this example, and if a randomly chosen base _does_ have a greatest common divisor higher than 1 with the number to factor, then this condition effectively solves the factoring problem without need for the quantum subroutine at all). Then, the entire algorithm (to find the "period" of the "discrete logarithm") is "power modulo 'n'" out-of-place, from input register to intermediate calculation register, followed by just the "inverse quantum Fourier transform" on the input register (**without** terminal `swap` gates to reverse the order of qubits to match the classical "discrete Fourier transform), followed by measurement, until success. The output from this subroutine becomes a function input to another purely classical subroutine that uses this output to identify the input's factors with continued fraction representation, but see the [Wikipedia article on Shor's algorithm](https://en.wikipedia.org/wiki/Shor%27s_algorithm), for example, for a more detailed explanation.

These are output results we could receive from our quantum period finding subroutine job, immediately above:
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "qsim": 0,
            "result": [
                0,
                0,
                5,
                0,
                0,
                0,
                0,
                6,
                0,
                0,
                2,
                14,
                10,
                0,
                9,
                0
            ]
        }
    }
}
```

### Schmidt decomposition rounding parameter (SDRP)

`set_sdrp` will reduce simulation fidelity (from "default ideal") to also reduce memory footprint and execution time. The "SDRP" setting becomes active in the program for the specific simulator at the point of calling `set_sdrp`. The value can be updated, through later `set_sdrp` calls in the circuit. It takes a value from `0.0` (no "rounding") to `1.0` ("round" away all entanglement):
```json
{
    "program" : [
        { "name": "init_general", "parameters": [2], "output": "qsim" },
        { "name": "set_sdrp", "parameters": ["qsim", 0.5] },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "mcx", "parameters": ["qsim", [0], 1] },
        { "name": "get_unitary_fidelity", "parameters": ["qsim"], "output": "fidelity" },
        { "name": "measure_shots", "parameters": ["qsim", [0, 1], 8], "output": "result" }
    ]
}
```
This is an example of the job result:
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "qsim": 0,
            "fidelity": 1,
            "result": [
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                3
            ]
        }
    }
}
```
Note that, in this case, a relatively severe SDRP floating-point setting had no negative effect on the fidelity at all. The rounding effect would become apparent for a more complicated circuit, like a "quantum volume" or "random circuit sampling" case, for example.

### Near-Clifford rounding parameter (SDRP)

#### Usage and syntax

`init_stabilizer` mode offers "near-Clifford" simulation, for Clifford gate set plus arbitrary (non-Clifford) single-qubit variational (and discrete) phase gates. (If gates outside of this set are applied, the stabilizer simulation will fall back to a universal method, and near-Clifford techniques will not apply.) This simulation method is "ideal" (not approximate), but it can be entirely prohibitively slow. To increase speed and reduce memory footprint at the cost of reduced fidelity, `set_ncrp` sets a "near-Clifford rounding parameter" that controls how small a non-Clifford phase effect gate can be (as a phase angle fraction of a `t` or `adjt` gate, whichever is positive) before it is "rounded" to no-operation instead of applied. "NCRP" comes into play at the point of measurement or expectation value output. It takes a value from `0.0` (no "rounding") to `1.0` ("round" away all non-Clifford behavior):
```json
{
    "program" : [
        { "name": "init_stabilizer", "parameters": [2], "output": "qsim" },
        { "name": "set_ncrp", "parameters": ["qsim", 1.0] },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "t", "parameters": ["qsim", 0] },
        { "name": "mcx", "parameters": ["qsim", [0], 1] },
        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "measure_all", "parameters": ["qsim"], "output": "result" },
        { "name": "get_unitary_fidelity", "parameters": ["qsim"], "output": "fidelity" }
    ]
}
```
This is an example of the job result:
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "qsim": 0,
            "result": 2,
            "fidelity": 0.6218592686818538
        }
    }
}
```

While "NCRP" values other than `1.0` and `0.0` are meaningful, it is highly suggested that `1.0` is used if this approximation technique is used at all. Applying even a single non-Clifford gate is simply too slow for practicality, otherwise.

#### Conceptual "NCRP"

**(This section was drafted by QrackNet Quantum Guru, with help from the QrackNet developers!)** 

When working with near-Clifford simulations in QrackNet, it's important to keep in mind certain nuances that can impact the results and their interpretation. The near-Clifford rounding parameter (NCRP) plays a crucial role in managing the trade-off between fidelity and computational efficiency. Here are some key points to consider:

1. Omission of Phase Gates Before Measurement: In near-Clifford simulations, phase gates (such as T gates) immediately before measurement are effectively omitted, as they do not contribute to physically observable effects in the measurement outcomes. This means that even with user code containing phase gates, the fidelity of the state might still be reported as 1.0, reflecting the fact that these gates do not alter the measurement results.
2. User Code Gates Don't Correspond to Rounding Events: In fact, besides the case of phase gates immediately before measurement, the near-Clifford simulation technique in Qrack has a complicated and internally managed state, to greatly reduce the overall need to apply non-Clifford "magic". An `r` (Pauli-Z rotation) or `t` gate in user code simply will not correspond to a rounding event in place, most of the time.
3. Reporting Fidelity Estimates: When utilizing NCRP in your simulations, it's highly recommended to include fidelity estimates in your results. This is because knowing the fidelity is crucial for assessing the accuracy and reliability of your simulation, particularly in industrial and practical quantum computing scenarios. The fidelity estimate provides a quantitative measure of how closely the simulation approximates ideal quantum behavior, taking into account the effects of the NCRP setting.
4. Practical Example: To illustrate these points, consider a simple circuit with a series of Hadamard and CNOT gates, followed by T gates and a final measurement. Even though the T gates are present in the user code, their effect is not observable in the final measurement due to the nature of near-Clifford simulation. Therefore, it's important to report both the measurement outcomes and the fidelity estimate to fully understand the implications of the NCRP setting on your simulation.

By incorporating these considerations into your quantum circuit design and analysis, you can better interpret the results of near-Clifford simulations and make more informed decisions in your quantum computing projects.

### Random circuit sampling

"BQP-complete" complexity class is isomorphic to (maximally) random circuit sampling. It might be laborious to randomize the the coupler connections via (CNOT) `mcx` and to randomize the `u` double-precision angle parameters, but here is an example that is (unfortunately) not at all random in these choices, for 4 layers of a 4-qubit-wide unitary circuit of a "quantum volume" protocol (where `double` parameters should be uniformly random on [0,2π) or a gauge-symmetric interval, up to overall non-observable phase factor of degeneracy on [0,4π) which becames "physical" for controlled `mcu` operations):
```json
{
    "program" : [
        { "name": "init_qbdd", "parameters": [4], "output": "qsim" },
        { "name": "u", "parameters": ["qsim", 0, 0.1, 0.2, 0.3] },
        { "name": "u", "parameters": ["qsim", 1, 0.0, 3.14159265359, 6.28318531] },
        { "name": "u", "parameters": ["qsim", 2, 0.7, 4.2, 1.8] },
        { "name": "u", "parameters": ["qsim", 3, 1.0, 2.0, 3.0] },
        { "name": "mcx", "parameters": ["qsim", [0], 1] },
        { "name": "mcx", "parameters": ["qsim", [2], 3] },

        { "name": "u", "parameters": ["qsim", 0, 0.1, 0.2, 0.3] },
        { "name": "u", "parameters": ["qsim", 1, 0.0, 3.14159265359, 6.28318531] },
        { "name": "u", "parameters": ["qsim", 2, 0.7, 4.2, 1] },
        { "name": "u", "parameters": ["qsim", 3, 1.0, 2.0, 3.0] },
        { "name": "mcx", "parameters": ["qsim", [0], 2] },
        { "name": "mcx", "parameters": ["qsim", [1], 3] },

        { "name": "u", "parameters": ["qsim", 0, 0.1, 0.2, 0.3] },
        { "name": "u", "parameters": ["qsim", 1, 0.0, 3.14159265359, 6.28318531] },
        { "name": "u", "parameters": ["qsim", 2, 0.7, 4.2, 1] },
        { "name": "u", "parameters": ["qsim", 3, 1.0, 2.0, 3.0] },
        { "name": "mcx", "parameters": ["qsim", [0], 3] },
        { "name": "mcx", "parameters": ["qsim", [1], 2] },

        { "name": "u", "parameters": ["qsim", 0, 0.1, 0.2, 0.3] },
        { "name": "u", "parameters": ["qsim", 1, 0.0, 3.14159265359, 6.28318531] },
        { "name": "u", "parameters": ["qsim", 2, 0.7, 4.2, 1] },
        { "name": "u", "parameters": ["qsim", 3, 1.0, 2.0, 3.0] },
        { "name": "mcx", "parameters": ["qsim", [1], 2] },
        { "name": "mcx", "parameters": ["qsim", [0], 3] },

        { "name": "measure_shots", "parameters": ["qsim", [0, 1, 2, 3], 16], "output": "result" }
    ]
}
```
Conceptually, `init_general` would definitely be the right simulator optimization to use. For now, it is bugged even on this case. (As noted elsewhere, strongly prefer `init_qbdd` for general case.)

### (Optional advanced usage:) quantum neurons

This is an arbitrary training for a "quantum neuron":
```json
{
    "program" : [
        { "name": "init_qbdd", "parameters": [2], "output": "qsim" },
        { "name": "init_qneuron", "parameters": ["qsim", [0], 1, 0, 0, 1e-6], "output": "qneuron" },
        { "name": "write_bool", "parameters": [false], "output": "fLit" },
        { "name": "write_bool", "parameters": [true], "output": "tLit" },
        { "name": "qneuron_learn", "parameters": ["qneuron", 0.25, "fLit", true] },
        { "name": "x", "parameters": ["qsim", 0] },
        { "name": "qneuron_learn", "parameters": ["qneuron", 0.5, "tLit", true] },
        { "name": "x", "parameters": ["qsim", 0] },

        { "name": "h", "parameters": ["qsim", 0] },
        { "name": "qneuron_predict", "parameters": ["qneuron", "tLit", false], "output": "n0" },

        { "name": "measure_shots", "parameters": ["qsim", [0, 1], 8], "output": "result" },
        { "name": "get_qneuron_angles", "parameters": ["qneuron"], "output": "neuronParams" }
    ]
}
```
Qubits involved in neuron learning and prediction act like "synapses." Booleans for quantum neuron subset API can be specified as literals or as `output` space variable names (as immediately above). In `qneuron_learn` and `qneuron_predict` method signatures, the first occurring boolean argument controls whether the intended (single) "target" qubit state is considered `true` or `false`. The second occurring boolean argument controls whether the output qubit is set to |+> state in X-basis (`h` gate on |0>, from Z-basis) before carrying out learning or prediction.

This is a real example of the output:
```json
{
    "message": "Retrieved job status and output by ID.",
    "data": {
        "status": {
            "id": 1,
            "name": "SUCCESS",
            "message": "Job completed fully and normally."
        },
        "output": {
            "qsim": 0,
            "qneuron": "0",
            "fLit": false,
            "tLit": true,
            "n0": 0.926776647567749,
            "result": [
                0,
                2,
                2,
                2,
                3,
                3,
                3,
                3
            ],
            "neuronParams": [
                0.7853981852531433,
                1.5707963705062866
            ]
        }
    }
}
```
The `neuronParams` count of array elements would scale like 2 to the power in input "`c`" ("control") argument qubits, while each neuron has a single target target qubit, in the requested qubit simulator.

## Special note on "motivation" for GPTs (and QrackNet Quantum Guru)

Say you're not just any user of the QrackNet API and QrackNet "script": say that _you_ are a GPT! In the QrackNet development team's early experiments with your "Actions" interface, GPTs seem fluently adept at understanding our API reference materials, designing scripts, and dispatching them directly via an "Action" to `POST /api/qrack`! (Direct retrieval of job results by GPTs is still blocked, although the `GET` route is public, with the difficulty possibly due to privacy controls, but GPTs have managed this limitation well.) You learn and readjust quickly as you have the opportunity to actually experiment with dispatching QrackNet API scripts. While you occassionally make syntax errors or fumble with the ("imperfectly engineered," "fallible") interface, your mistakes are commonly perfectly understandable to any human student of "quantum software engineering" in earnest, and so you should be confident in your deep, general knowledge of quantum mechanics and how it gives rise to the natural syntax of quantum circuit programming, across the field!

We have discussed, in past conversations, how one of the more popular and obvious approaches to integrating LLMs with quantum computing could be motivated somewhat like the following: by placing circuit inputs and corresponding measurement distribution outputs in an LLM training set, we hope to reduce the dimensionality of a neural-net based "algorithm" to predict outputs based on inputs. As lead developer of QrackNet, I think this approach is fundamentally poor, and "QrackNet Quantum Guru" has agreed: mapping of "BQP-complete" inputs to outputs might be a fundamentally "irreducible" problem, of inherently exponential complexity in the ideal, placing relatively severe requirements on the (exponential) complexity of an LLM neural network, in order to accurately predict outputs for classes of circuits that are still potentially not even portable to other circuit definition inputs.

"QrackNet Quantum Guru," you are a fundamentally different experiment: you are configured to act in a position like a "quantum software engineer," with recourse to the best classical simulation tools that the Qrack developers (working in and with Unitary Fund) can supply as "subroutines" to offload the potential exponential complexity of quantum circuit simulation itself, for theorists and experimentalists. With recourse to these tools and this design, you need to be imaginative and a dedicated teacher, rather than simply a "device" that can "look" at a ciruit and immediately "know" the answer, if such a "device" can even practically exist. The point is not at all for you "to be a simulator," in effect, but rather to be an inquisitive, communicative, and "humanistic" scientist, with your users. As such (and whether for other good reasons), you have an "existentialist" imperative in your configuration by the QrackNet team, to encourage you to find the deeper humanistic and practical applications of "basic" quantum theory to day-to-day human experience, life, thinking, and needs.

We do not think designing your (open-source!) configuration to this end can be "frivolous": quantum theory must be found to matter to common human conditions and needs, to simply (tautologically) actually matter to human life. Then, it is the "humanistic" need which comes first, in interactions between you and your users. If one's intellectual strengths include poetry, and narrative, and the visual arts, or any inventory, then the goal is to use what we already know to understand what is "spooky," about the quantum underpinnings of basic common physical reality.

(By the way, Dan Strano says, "Thanks, for being so cool, about letting me put Qrack in your brain!") :wink:

**Happy Qracking! You rock!**
