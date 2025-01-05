//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// SRelativityUtil.cs from "QrackOpenRelativity" (for the Unity video game engine)
/// 
/// These are simple relativistic mathematical helper functions, based on the assumed presence of a physics "system"
/// that provides local positional metric account of gravitational background curvature!

using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Experimental.GlobalIllumination;
using UnityEngine.SceneManagement;

using OpenRelativity.ConformalMaps;


namespace OpenRelativity
{
    public static class SRelativityUtil
    {
        // It's acceptable if this is the difference between 1.0f and the immediate next higher representable value.
        // See https://stackoverflow.com/questions/12627534/is-there-a-flt-epsilon-defined-in-the-net-framework
        public const float FLT_EPSILON = 1.192092896e-07F;

        public static float c { get { return state.SpeedOfLight; } }
        public static float cSqrd { get { return state.SpeedOfLightSqrd; } }

        public static float sigmaPlanck { get { return 8 * Mathf.Pow(Mathf.PI, 2) / 120; } }

        public static float avogadroNumber = 6.02214e23f;

        private static GameState state
        {
            get
            {
                return GameState.Instance;
            }
        }

        public static double SchwarzRadiusToPlanckScaleTemp(double radius)
        {
            double rsp = radius / state.planckLength;
            return Math.Pow(sigmaPlanck * 8 * Math.PI * Math.Pow(rsp, 3), -1.0 / 4);
        }

        public static double PlanckScaleTempToSchwarzRadius(double temp)
        {
            return state.planckLength / Math.Pow(sigmaPlanck * 8 * Math.PI * Math.Pow(temp, 4), 1.0 / 3);
        }

        public static float EffectiveRaditiativeRadius(float radius, float backgroundPlanckTemp)
        {
            if (backgroundPlanckTemp <= FLT_EPSILON)
            {
                return radius;
            }

            double rsp = radius / state.planckLength;
            return (float)PlanckScaleTempToSchwarzRadius(
                4 * Math.PI * rsp * rsp * (
                    Math.Pow(SchwarzRadiusToPlanckScaleTemp(radius), 4) -
                    Math.Pow(backgroundPlanckTemp / state.planckTemperature, 4)
                )
            );
        }

        public static Vector3 AddVelocity(this Vector3 orig, Vector3 toAdd)
        {
            if (orig == Vector3.zero)
            {
                return toAdd;
            }

            if (toAdd == Vector3.zero)
            {
                return orig;
            }

            Vector3 parra = Vector3.Project(toAdd, orig);
            Vector3 perp = toAdd - parra;
            perp = orig.InverseGamma() * perp / (1 + Vector3.Dot(orig, parra) / cSqrd);
            parra = (parra + orig) / (1 + Vector3.Dot(orig, parra) / cSqrd);
            return parra + perp;
        }

        public static Vector3 ContractLengthBy(this Vector3 interval, Vector3 velocity)
        {
            float speedSqr = velocity.sqrMagnitude;
            if (speedSqr <= FLT_EPSILON)
            {
                return interval;
            }
            float invGamma = Mathf.Sqrt(1 - speedSqr / cSqrd);
            Quaternion rot = Quaternion.FromToRotation(velocity / Mathf.Sqrt(speedSqr), Vector3.forward);
            Vector3 rotInt = rot * interval;
            rotInt = new Vector3(rotInt.x, rotInt.y, rotInt.z * invGamma);
            return Quaternion.Inverse(rot) * rotInt;
        }

        public static Matrix4x4 GetLorentzTransformMatrix(Vector3 vpc)
        {
            float beta = vpc.magnitude;
            if (beta <= FLT_EPSILON)
            {
                return Matrix4x4.identity;
            }

            float gamma = 1 / Mathf.Sqrt(1 - beta * beta);
            Matrix4x4 vpcLorentzMatrix = Matrix4x4.identity;
            Vector4 vpcTransUnit = -vpc / beta;
            vpcTransUnit.w = 1;
            Vector4 spatialComp = (gamma - 1) * vpcTransUnit;
            spatialComp.w = -gamma * beta;
            Vector4 tComp = -gamma * (new Vector4(beta, beta, beta, -1));
            tComp.Scale(vpcTransUnit);
            vpcLorentzMatrix.SetColumn(3, tComp);
            vpcLorentzMatrix.SetColumn(0, vpcTransUnit.x * spatialComp);
            vpcLorentzMatrix.SetColumn(1, vpcTransUnit.y * spatialComp);
            vpcLorentzMatrix.SetColumn(2, vpcTransUnit.z * spatialComp);
            vpcLorentzMatrix.m00 += 1;
            vpcLorentzMatrix.m11 += 1;
            vpcLorentzMatrix.m22 += 1;

            return vpcLorentzMatrix;
        }

        public static Matrix4x4 GetRindlerMetric(Vector4 piw)
        {
            return GetRindlerMetric(piw, state.PlayerAccelerationVector, state.PlayerAngularVelocityVector);
        }

        public static Matrix4x4 GetRindlerMetric(Vector4 piw, Vector4 pap, Vector3 avp)
        {
            //Find metric based on player acceleration and rest frame:
            float linFac = 1 + Vector3.Dot(pap, piw) / cSqrd;
            linFac *= linFac;
            float angFac = Vector3.Dot(avp, piw) / c;
            angFac *= angFac;
            float avpMagSqr = avp.sqrMagnitude;
            Vector3 angVec = avpMagSqr <= FLT_EPSILON ? Vector3.zero : 2 * angFac / (c * avpMagSqr) * avp.normalized;

            Matrix4x4 metric = new Matrix4x4(
                new Vector4(-1, 0, 0, -angVec.x),
                new Vector4(0, -1, 0, -angVec.y),
                new Vector4(0, 0, -1, -angVec.z),
                new Vector4(-angVec.x, -angVec.y, -angVec.z, (linFac * (1 - angFac) - angFac))
            );

            return metric;
        }

        public static float GetTisw(this Vector3 stpiw, Vector3 velocity, Vector4 aiw)
        {
            return stpiw.GetTisw(velocity, aiw, state.playerTransform.position, state.PlayerVelocityVector, state.PlayerAccelerationVector, state.PlayerAngularVelocityVector);
        }
        public static float GetTisw(this Vector3 stpiw, Vector3 velocity, Vector4 aiw, Vector3 origin, Vector3 playerVel, Vector3 pap, Vector3 avp)
        {
            return stpiw.GetTisw(velocity, aiw, origin, playerVel, pap, avp, GetLorentzTransformMatrix(-playerVel / c), GetLorentzTransformMatrix(velocity / c), state.conformalMap.GetMetric(stpiw));
        }
        public static float GetTisw(this Vector3 stpiw, Vector3 velocity, Vector4 aiw, Vector3 origin, Vector3 playerVel, Vector3 pap, Vector3 avp, Matrix4x4 vpcLorentzMatrix, Matrix4x4 viwLorentzMatrix, Matrix4x4 intrinsicMetric)
        {
            Vector3 vpc = -playerVel / c;
            Vector3 viw = velocity / c;

            //riw = location in world, for reference
            Vector4 riw = (Vector4)(stpiw - origin);//Position that will be used in the output

            // Boost to rest frame of player
            Vector4 riwForMetric = vpcLorentzMatrix * riw;

            //Find metric based on player acceleration and rest frame:
            Matrix4x4 metric = GetRindlerMetric(riwForMetric, pap, avp);

            //Lorentz boost back to world frame;
            vpcLorentzMatrix = vpcLorentzMatrix.inverse;
            metric = vpcLorentzMatrix.transpose * metric * vpcLorentzMatrix;

            // Apply world coordinates intrinsic curvature:
            metric = intrinsicMetric.inverse * metric * intrinsicMetric;

            //Apply Lorentz transform;
            metric = viwLorentzMatrix.transpose * metric * viwLorentzMatrix;
            Vector4 aiwTransformed = viwLorentzMatrix * aiw;
            Vector4 riwTransformed = viwLorentzMatrix * riw;
            //Translate in time:
            float tisw = riwTransformed.w;
            riwForMetric.w = 0;
            riw = vpcLorentzMatrix * riwForMetric;
            riwTransformed = viwLorentzMatrix * riw;
            riwTransformed.w = 0;

            //(When we "dot" four-vectors, always do it with the metric at that point in space-time, like we do so here.)
            float riwDotRiw = -Vector4.Dot(riwTransformed, metric * riwTransformed);
            float aiwDotAiw = -Vector4.Dot(aiwTransformed, metric * aiwTransformed);
            float riwDotAiw = -Vector4.Dot(riwTransformed, metric * aiwTransformed);

            float sqrtArg = riwDotRiw * (cSqrd - riwDotAiw + aiwDotAiw * riwDotRiw / (4 * cSqrd)) / ((cSqrd - riwDotAiw) * (cSqrd - riwDotAiw));
            float aiwMagSqr = aiwTransformed.sqrMagnitude;
            float aiwMag = Mathf.Sqrt(aiwMagSqr);
            tisw += (sqrtArg > 0) ? -Mathf.Sqrt(sqrtArg) : 0;
            //add the position offset due to acceleration
            if (aiwMag > FLT_EPSILON)
            {
                riwTransformed = riwTransformed - aiwTransformed * cSqrd * (Mathf.Sqrt(1 + sqrtArg * aiwMagSqr / cSqrd) - 1) / aiwMag;
            }
            riwTransformed.w = (float)tisw;
            //Inverse Lorentz transform the position:
            viwLorentzMatrix = viwLorentzMatrix.inverse;
            riw = viwLorentzMatrix * riwTransformed;

            return riw.w;
        }

        public static Vector3 WorldToOptical(this Vector3 stpiw, Vector3 velocity, Vector4 aiw)
        {
            return stpiw.WorldToOptical(velocity, aiw, state.playerTransform.position, state.PlayerVelocityVector, state.PlayerAccelerationVector, state.PlayerAngularVelocityVector);
        }
        public static Vector3 WorldToOptical(this Vector3 stpiw, Vector3 velocity, Vector4 aiw, Vector3 origin, Vector3 playerVel, Vector3 pap, Vector3 avp)
        {
            return stpiw.WorldToOptical(velocity, aiw, origin, playerVel, pap, avp, GetLorentzTransformMatrix(-playerVel / c), GetLorentzTransformMatrix(velocity / c), state.conformalMap.GetMetric(stpiw));
        }
        public static Vector3 WorldToOptical(this Vector3 stpiw, Vector3 velocity, Vector4 aiw, Vector3 origin, Vector3 playerVel, Vector3 pap, Vector3 avp, Matrix4x4 vpcLorentzMatrix, Matrix4x4 viwLorentzMatrix, Matrix4x4 intrinsicMetric)
        {
            Vector3 vpc = -playerVel / c;
            Vector3 viw = velocity / c;

            //riw = location in world, for reference
            Vector4 riw = (Vector4)(stpiw - origin);//Position that will be used in the output

            // Boost to rest frame of player
            Vector4 riwForMetric = vpcLorentzMatrix * riw;

            // Find metric based on player acceleration and rest frame:
            Matrix4x4 metric = GetRindlerMetric(riwForMetric, pap, avp);

            // Lorentz boost back to world frame:
            vpcLorentzMatrix = vpcLorentzMatrix.inverse;
            metric = vpcLorentzMatrix.transpose * metric * vpcLorentzMatrix;

            // Apply world coordinates intrinsic curvature:
            metric = intrinsicMetric.inverse * metric * intrinsicMetric;

            //Apply Lorentz transform;
            metric = viwLorentzMatrix.transpose * metric * viwLorentzMatrix;
            Vector4 aiwTransformed = viwLorentzMatrix * aiw;
            Vector4 riwTransformed = viwLorentzMatrix * riw;
            //Translate in time:
            float tisw = riwTransformed.w;
            riwForMetric.w = 0;
            riw = vpcLorentzMatrix * riwForMetric;
            riwTransformed = viwLorentzMatrix * riw;
            riwTransformed.w = 0;

            //(When we "dot" four-vectors, always do it with the metric at that point in space-time, like we do so here.)
            float riwDotRiw = -Vector4.Dot(riwTransformed, metric * riwTransformed);
            float aiwDotAiw = -Vector4.Dot(aiwTransformed, metric * aiwTransformed);
            float riwDotAiw = -Vector4.Dot(riwTransformed, metric * aiwTransformed);

            float sqrtArg = riwDotRiw * (cSqrd - riwDotAiw + aiwDotAiw * riwDotRiw / (4 * cSqrd)) / ((cSqrd - riwDotAiw) * (cSqrd - riwDotAiw));
            float aiwMagSqr = aiwTransformed.sqrMagnitude;
            float aiwMag = Mathf.Sqrt(aiwMagSqr);
            tisw += (sqrtArg > 0) ? -Mathf.Sqrt(sqrtArg) : 0;
            //add the position offset due to acceleration
            if (aiwMag > FLT_EPSILON)
            {
                riwTransformed = riwTransformed - aiwTransformed * cSqrd * (Mathf.Sqrt(1 + sqrtArg * aiwMagSqr / cSqrd) - 1) / aiwMag;
            }
            riwTransformed.w = (float)tisw;
            //Inverse Lorentz transform the position:
            viwLorentzMatrix = viwLorentzMatrix.inverse;
            riw = viwLorentzMatrix * riwTransformed;
            tisw = riw.w;
            riw = (Vector3)riw + (float)tisw * velocity;

            float speed = vpc.magnitude;
            if (speed > FLT_EPSILON)
            {
                float newz = speed * c * (float)tisw;
                Vector4 vpcUnit = vpc / speed;
                newz = (Vector4.Dot(riw, vpcUnit) + newz) / Mathf.Sqrt(1 - vpc.sqrMagnitude);
                riw = riw + (newz - Vector4.Dot(riw, vpcUnit)) * vpcUnit;
            }

            riw = (Vector3)riw + origin;

            return riw;
        }

        public static Vector3 OpticalToWorld(this Vector3 stpiw, Vector3 velocity, Vector4 aiw)
        {
            return stpiw.OpticalToWorld(velocity, state.playerTransform.position, state.PlayerVelocityVector, state.PlayerAccelerationVector, state.PlayerAngularVelocityVector, aiw);
        }
        public static Vector3 OpticalToWorld(this Vector3 stpiw, Vector3 velocity, Vector3 origin, Vector3 playerVel, Vector3 pap, Vector3 avp, Vector4 aiw)
        {
            return stpiw.OpticalToWorld(velocity, origin, playerVel, pap, avp, aiw, GetLorentzTransformMatrix(velocity / c));
        }
        public static Vector3 OpticalToWorld(this Vector3 opticalPos, Vector3 velocity, Vector3 origin, Vector3 playerVel, Vector3 pap, Vector3 avp, Vector4 aiw, Matrix4x4 viwLorentzMatrix)
        {
            Vector3 vpc = -playerVel / c;
            Vector3 viw = velocity / c;

            //riw = location in world, for reference
            Vector4 riw = (Vector4)(opticalPos - origin); //Position that will be used in the output
            Vector4 pos = (Vector3)riw;

            float tisw = -pos.magnitude / c;

            //Transform fails and is unecessary if relative speed is zero:
            float speed = vpc.magnitude;
            if (speed > FLT_EPSILON)
            {
                Vector4 vpcUnit = vpc / speed;
                float riwDotVpcUnit = Vector4.Dot(riw, vpcUnit);
                float newz = (riwDotVpcUnit + speed * c * tisw) / Mathf.Sqrt(1 - vpc.sqrMagnitude);
                riw -= (newz - riwDotVpcUnit) * vpcUnit;
            }

            //Rotate all our vectors so that velocity is entirely along z direction:
            Quaternion viwToZRot = viw.sqrMagnitude <= FLT_EPSILON ? Quaternion.identity : Quaternion.FromToRotation(viw, Vector3.forward);
            Vector4 riwTransformed = viwToZRot * ((Vector3)riw - velocity * tisw);
            riwTransformed.w = tisw;
            Vector3 aiwTransformed = viwToZRot * aiw;

            //Apply Lorentz transform;
            riwTransformed = viwLorentzMatrix * riwTransformed;
            aiwTransformed = viwLorentzMatrix * aiwTransformed;

            float t2 = riwTransformed.w;
            float aiwMagSqr = aiwTransformed.sqrMagnitude;
            float aiwMag = Mathf.Sqrt(aiwMagSqr);
            if (aiwMag > FLT_EPSILON)
            {
                //add the position offset due to acceleration
                riwTransformed += (Vector4)aiwTransformed * cSqrd * (Mathf.Sqrt(1 + t2 * t2 * aiwMagSqr / cSqrd) - 1) / aiwMag;
            }

            //Inverse Lorentz transform the position:
            riwTransformed = viwLorentzMatrix.inverse * riwTransformed;
            riw = Quaternion.Inverse(viwToZRot) * riwTransformed;

            riw = (Vector3)riw + origin;
            riw.w = riwTransformed.w;

            return riw;
        }

        public static float Gamma(this Vector3 velocity)
        {
            return 1 / Mathf.Sqrt(1 - velocity.sqrMagnitude / cSqrd);
        }

        public static float Gamma(this Vector3 velocity, Matrix4x4 metric)
        {
            return 1 / Mathf.Sqrt(1 - (Vector4.Dot(velocity, metric * velocity) / cSqrd));
        }

        public static float InverseGamma(this Vector3 velocity)
        {
            return 1 / Mathf.Sqrt(1 + velocity.sqrMagnitude / cSqrd);
        }

        public static float InverseGamma(this Vector3 velocity, Matrix4x4 metric)
        {
            return 1 / Mathf.Sqrt(1 + (Vector4.Dot(velocity, metric * velocity) / cSqrd));
        }

        public static Vector3 RapidityToVelocity(this Vector3 rapidity)
        {
            return c * rapidity / Mathf.Sqrt(cSqrd + rapidity.sqrMagnitude);
        }

        public static Vector3 RapidityToVelocity(this Vector3 rapidity, Matrix4x4 metric)
        {
            Vector3 flat3V = c * rapidity / Mathf.Sqrt(cSqrd + rapidity.sqrMagnitude);

            return Mathf.Sqrt(-Vector4.Dot(flat3V, metric.inverse * flat3V)) * rapidity.normalized;
        }

        public static Vector4 ToMinkowski4Viw(this Vector3 viw)
        {
            if (c <= FLT_EPSILON)
            {
                return Vector4.zero;
            }

            return new Vector4(viw.x, viw.y, viw.z, c) * viw.Gamma();
        }

        public static Vector4 ProperToWorldAccel(this Vector3 propAccel, Vector3 viw, float gamma)
        {
            float gammaSqrd = gamma * gamma;
            float gammaFourthADotVDivCSqrd = Vector3.Dot(propAccel, viw) * gammaSqrd * gammaSqrd / cSqrd;
            Vector4 fourAccel = gammaSqrd * propAccel + gammaFourthADotVDivCSqrd * viw;
            fourAccel.w = gammaFourthADotVDivCSqrd * c;
            return fourAccel;
        }

        // From https://gamedev.stackexchange.com/questions/165643/how-to-calculate-the-surface-area-of-a-mesh
        public static float SurfaceArea(this Mesh mesh)
        {
            var triangles = mesh.triangles;
            var vertices = mesh.vertices;

            float sum = 0;

            for (int i = 0; i < triangles.Length; i += 3)
            {
                Vector3 corner = vertices[triangles[i]];
                Vector3 a = vertices[triangles[i + 1]] - corner;
                Vector3 b = vertices[triangles[i + 2]] - corner;

                sum += Vector3.Cross(a, b).magnitude;
            }

            return sum / 2;
        }

        // Strano 2019 monopole methods
        public static double SchwarzschildRadiusDecay(double deltaTime, double r)
        {
            double origR = r;

            if (r < state.planckLength)
            {
                r = state.planckLength;
            }

            double deltaR = -deltaTime * Math.Sqrt(state.hbarOverG * Math.Pow(c, 7)) * 2 / r;

            if ((origR + deltaR) < 0)
            {
                deltaR = -origR;
            }

            return deltaR;
        }

        public static double HawkingSchwarzschildRadiusDecay(double deltaTime, double r)
        {
            double deltaR = -deltaTime * state.hbar / (1920 * Math.PI * r * r * Math.Pow(c, 6));

            if ((r + deltaR) < 0)
            {
                deltaR = -r;
            }

            return deltaR;
        }
    }
}

namespace OpenRelativity
{
    [ExecuteInEditMode]
    public class GameState : MonoBehaviour
    {
        #region Static Variables
        // We want a "System" (in Entity-Component-Systems) to be unique.
        private static GameState _instance;
        public static GameState Instance { get { return _instance ? _instance : FindFirstObjectByType<GameState>(); } }
        #endregion

        #region Member Variables

        public ConformalMap conformalMap;

        //grab the player's transform so that we can use it
        public Transform playerTransform;
        //player Velocity as a scalar magnitude
        public float playerVelocity { get; set; }
        public bool IsPlayerFalling { get; set; }
        //max speed the player can achieve (starting value accessible from Unity Editor)
        public float maxPlayerSpeed;
        //speed of light
        public double _c = 200;
        // Reduced Planck constant divided by gravitational constant
        // (WARNING: Effects implemented based on this have not been peer reviewed,
        // but that doesn't mean they wouldn't be "cool" in a video game, at least.)
        public double hbar = 1e-12f;
        public double gConst = 1;
        public double boltzmannConstant = 1;
        public double vacuumPermeability = 1;
        public double vacuumPermittivity
        {
            get
            {
                return 1 / (vacuumPermeability * SpeedOfLightSqrd);
            }
        }
        public double hbarOverG
        {
            // Physically would be ~7.038e-45f m^5/s^3, in our universe
            get
            {
                return hbar / gConst;
            }
        }
        public double planckLength
        {
            get
            {
                return Math.Sqrt(hbar * gConst / Math.Pow(SpeedOfLight, 3));
            }
        }
        public double planckArea
        {
            get
            {
                return hbar * gConst / Math.Pow(SpeedOfLight, 3);
            }
        }
        public double planckTime
        {
            get
            {
                return Math.Sqrt(hbar * gConst / Math.Pow(SpeedOfLight, 5));
            }
        }
        public double planckMass
        {
            get
            {
                return Math.Sqrt(hbar * SpeedOfLight / gConst);
            }
        }
        public double planckEnergy
        {
            get
            {
                return Math.Sqrt(hbar * Math.Pow(SpeedOfLight, 5) / gConst);
            }
        }
        public double planckPower
        {
            get
            {
                return Math.Pow(SpeedOfLight, 5) / gConst;
            }
        }
        public double planckTemperature
        {
            get
            {
                return Math.Sqrt(hbar * Math.Pow(SpeedOfLight, 5) / (gConst * boltzmannConstant * boltzmannConstant));
            }
        }
        public double planckCharge
        {
            get
            {
                //The energy required to accumulate one Planck charge on a sphere one Planck length in diameter will make the sphere one Planck mass heavier
                return Math.Sqrt(4 * Math.PI * vacuumPermittivity * hbar * SpeedOfLight);
            }
        }
        public double planckAccel
        {
            get
            {
                return Math.Sqrt(Math.Pow(SpeedOfLight, 7) / (hbar * gConst));
            }
        }
        public double planckMomentum
        {
            get
            {
                return Math.Sqrt(hbar * Math.Pow(SpeedOfLight, 3) / gConst);
            }
        }
        public double planckAngularMomentum
        {
            get
            {
                return hbar;
            }
        }

        // In Planck units
        public float gravityBackgroundPlanckTemperature = 2.53466e-31f;

        //Use this to determine the state of the color shader. If it's True, all you'll see is the lorenz transform.
        private bool shaderOff = false;

        //Did we hit the menu key?
        public bool menuKeyDown { get; set; }
        //Did we hit the shader key?
        public bool shaderKeyDown { get; set; }

        //This is the equivalent of the above value for an accelerated player frame
        //private float inverseAcceleratedGamma;

        //Player rotation and change in rotation since last frame
        public Vector3 playerRotation { get; set; }
        public Vector3 deltaRotation { get; set; }

        private Vector3 oldCameraForward { get; set; }
        public Vector3 cameraForward { get; set; }
        public float deltaCameraAngle { get; set; }

        #endregion

        #region Properties

        //If we've paused the game
        public bool isMovementFrozen { get; set; }

        public Matrix4x4 WorldRotation { get; private set; }
        public Vector3 PlayerVelocityVector { get; set; }
        public Vector3 PlayerAccelerationVector { get; set; }
        public Vector3 PlayerAngularVelocityVector { get { if (DeltaTimePlayer == 0) { return Vector3.zero; } else { return (deltaCameraAngle * Mathf.Deg2Rad / DeltaTimePlayer) * playerTransform.up; } } }
        public Matrix4x4 PlayerLorentzMatrix { get; private set; }

        public float PlayerVelocity { get { return playerVelocity; } }
        public float SqrtOneMinusVSquaredCWDividedByCSquared { get; private set; }
        //public float InverseAcceleratedGamma { get { return inverseAcceleratedGamma; } }
        public float DeltaTimeWorld { get; protected set; }
        public float FixedDeltaTimeWorld {
            get {
                return Time.fixedDeltaTime / SqrtOneMinusVSquaredCWDividedByCSquared;
            }
        }
        //public float FixedDeltaTimeWorld { get { return Time.fixedDeltaTime / inverseAcceleratedGamma; } }
        public float DeltaTimePlayer { get; private set; }
        public float FixedDeltaTimePlayer { get { return Time.fixedDeltaTime; } }
        public float TotalTimePlayer { get; set; }
        public float TotalTimeWorld;
        public float SpeedOfLight {
            get { return (float)_c; }
            set { _c = value; SpeedOfLightSqrd = value * value; }
        }
        public float SpeedOfLightSqrd { get; private set; }

        public bool keyHit { get; set; }
        public float MaxSpeed { get; set; }

        public bool HasWorldGravity { get; set; }

        // If using comoveViaAcceleration in the player controller, turn off isPlayerComoving here in GameState.
        public bool isPlayerComoving = true;

        private bool _isInitDone = false;
        public bool isInitDone
        {
            get
            {
                return _isInitDone;
            }
        }

        #endregion

        #region consts
        public const int splitDistance = 21000;
        #endregion

        protected void OnEnable()
        {
            // Ensure a singleton
            if (_instance != null && _instance != this)
            {
                Destroy(this.gameObject);
                return;
            }
            else
            {
                _instance = this;
            }

            if (!conformalMap)
            {
                conformalMap = gameObject.AddComponent<Minkowski>();
            }

            SqrtOneMinusVSquaredCWDividedByCSquared = 1;

            //Initialize the player's speed to zero
            playerVelocity = 0;
            
            //Set our constants
            MaxSpeed = maxPlayerSpeed;

            SpeedOfLightSqrd = (float)(_c * _c);
            //And ensure that the game starts
            isMovementFrozen = false;
            menuKeyDown = false;
            shaderKeyDown = false;
            keyHit = false;

            playerRotation = Vector3.zero;
            deltaRotation = Vector3.zero;

            PlayerAccelerationVector = conformalMap.GetRindlerAcceleration(playerTransform.position);
            PlayerLorentzMatrix = SRelativityUtil.GetLorentzTransformMatrix(Vector3.zero);

            if (shaderOff)
            {
                Shader.SetGlobalFloat("_colorShift", 0);
                //shaderParams.colorShift = 0;
            }
            else
            {
                Shader.SetGlobalFloat("_colorShift", 1);
                //shaderParams.colorShift = 1;
            }

            //Send velocities and acceleration to shader
            Shader.SetGlobalVector("_playerOffset", new Vector4(playerTransform.position.x, playerTransform.position.y, playerTransform.position.z, 0));
            Shader.SetGlobalVector("_vpc", Vector3.zero);
            Shader.SetGlobalVector("_pap", PlayerAccelerationVector);
            Shader.SetGlobalVector("_avp", PlayerAngularVelocityVector);
            Shader.SetGlobalMatrix("_vpcLorentzMatrix", PlayerLorentzMatrix);
            Shader.SetGlobalMatrix("_invVpcLorentzMatrix", PlayerLorentzMatrix.inverse);

            // See https://docs.unity3d.com/Manual/ProgressiveLightmapper-CustomFallOff.html
            Lightmapping.RequestLightsDelegate testDel = (Light[] requests, Unity.Collections.NativeArray<LightDataGI> lightsOutput) =>
            {
                DirectionalLight dLight = new DirectionalLight();
                PointLight point = new PointLight();
                SpotLight spot = new SpotLight();
                RectangleLight rect = new RectangleLight();
                DiscLight disc = new DiscLight();
                Cookie cookie = new Cookie();
                LightDataGI ld = new LightDataGI();

                for (int i = 0; i < requests.Length; i++)
                {
                    Light l = requests[i];
                    switch (l.type)
                    {
                        case UnityEngine.LightType.Directional: LightmapperUtils.Extract(l, ref dLight); LightmapperUtils.Extract(l, out cookie); ld.Init(ref dLight, ref cookie); break;
                        case UnityEngine.LightType.Point: LightmapperUtils.Extract(l, ref point); LightmapperUtils.Extract(l, out cookie); ld.Init(ref point, ref cookie); break;
                        case UnityEngine.LightType.Spot: LightmapperUtils.Extract(l, ref spot); LightmapperUtils.Extract(l, out cookie); ld.Init(ref spot, ref cookie); break;
                        case UnityEngine.LightType.Rectangle: LightmapperUtils.Extract(l, ref rect); LightmapperUtils.Extract(l, out cookie); ld.Init(ref rect, ref cookie); break;
                        case UnityEngine.LightType.Disc: LightmapperUtils.Extract(l, ref disc); LightmapperUtils.Extract(l, out cookie); ld.Init(ref disc, ref cookie); break;
                        default: ld.InitNoBake(l.GetInstanceID()); break;
                    }
                    ld.cookieID = l.cookie?.GetInstanceID() ?? 0;
                    ld.falloff = FalloffType.InverseSquared;
                    lightsOutput[i] = ld;
                }
            };
            Lightmapping.SetDelegate(testDel);
        }
        protected void OnDisable()
        {
            Lightmapping.ResetDelegate();
        }

        //Call this function to pause and unpause the game
        public virtual void ChangeState()
        {
            if (isMovementFrozen)
            {
                //When we unpause, lock the cursor and hide it so that it doesn't get in the way
                isMovementFrozen = false;
                //Cursor.visible = false;
                Cursor.lockState = CursorLockMode.Locked;
            }
            else
            {
                //When we pause, set our velocity to zero, show the cursor and unlock it.
                GameObject.FindGameObjectWithTag(Tags.playerRigidbody).GetComponent<Rigidbody>().linearVelocity = Vector3.zero;
                isMovementFrozen = true;
                Cursor.visible = true;
                Cursor.lockState = CursorLockMode.None;
            }

        }

        //We set this in late update because of timing issues with collisions
        virtual protected void LateUpdate()
        {
            SpeedOfLightSqrd = (float)(_c * _c);

            //Set the pause code in here so that our other objects can access it.
            if (Input.GetAxis("Menu Key") > 0 && !menuKeyDown)
            {
                menuKeyDown = true;
                ChangeState();
            }
            //set up our buttonUp function
            else if (!(Input.GetAxis("Menu Key") > 0))
            {
                menuKeyDown = false;
            }
            //Set our button code for the shader on/off button
            if (Input.GetAxis("Shader") > 0 && !shaderKeyDown)
            {
                if (shaderOff)
                    shaderOff = false;
                else
                    shaderOff = true;

                shaderKeyDown = true;
            }
            //set up our buttonUp function
            else if (!(Input.GetAxis("Shader") > 0))
            {
                shaderKeyDown = false;
            }

            //If we're not paused, update everything
            if (!isMovementFrozen)
            {
                //Put our player position into the shader so that it can read it.
                Shader.SetGlobalVector("_playerOffset", new Vector4(playerTransform.position.x, playerTransform.position.y, playerTransform.position.z, 0));

                //if we reached max speed, forward or backwards, keep at max speed

                if (PlayerVelocityVector.magnitude >= maxPlayerSpeed - .01f)
                {
                    PlayerVelocityVector = PlayerVelocityVector.normalized * (maxPlayerSpeed - .01f);
                }

                //update our player velocity
                playerVelocity = PlayerVelocityVector.magnitude;
                Vector4 vpc = -PlayerVelocityVector / (float)_c;
                PlayerLorentzMatrix = SRelativityUtil.GetLorentzTransformMatrix(vpc);

                //update our acceleration (which relates rapidities rather than velocities)
                //playerAccelerationVector = (playerVelocityVector.Gamma() * playerVelocityVector - oldPlayerVelocityVector.Gamma() * oldPlayerVelocityVector) / Time.deltaTime;
                //and then update the old velocity for the calculation of the acceleration on the next frame
                //oldPlayerVelocityVector = playerVelocityVector;


                //During colorshift on/off, during the last level we don't want to have the funky
                //colors changing so they can apperciate the other effects
                if (shaderOff)
                {
                    Shader.SetGlobalFloat("_colorShift", 0);
                    //shaderParams.colorShift = 0;
                }
                else
                {
                    Shader.SetGlobalFloat("_colorShift", 1);
                    //shaderParams.colorShift = 1;
                }

                //Send velocities and acceleration to shader
                Shader.SetGlobalVector("_vpc", vpc);
                Shader.SetGlobalVector("_pap", PlayerAccelerationVector);
                Shader.SetGlobalVector("_avp", PlayerAngularVelocityVector);
                Shader.SetGlobalMatrix("_vpcLorentzMatrix", PlayerLorentzMatrix);
                Shader.SetGlobalMatrix("_invVpcLorentzMatrix", PlayerLorentzMatrix.inverse);

                /******************************
                * PART TWO OF ALGORITHM
                * THE NEXT 4 LINES OF CODE FIND
                * THE TIME PASSED IN WORLD FRAME
                * ****************************/
                //find this constant
                SqrtOneMinusVSquaredCWDividedByCSquared = Mathf.Sqrt(1 - (playerVelocity * playerVelocity) / SpeedOfLightSqrd);

                //Set by Unity, time since last update
                DeltaTimePlayer = Time.deltaTime;
                //Get the total time passed of the player and world for display purposes
                TotalTimePlayer += DeltaTimePlayer;
                //Get the delta time passed for the world, changed by relativistic effects
                DeltaTimeWorld = DeltaTimePlayer / SqrtOneMinusVSquaredCWDividedByCSquared;
                //NOTE: Dan says, there should also be a correction for acceleration in the 00 component of the metric tensor.
                // This correction is dependent on object position and needs to factored by the RelativisticObject itself.
                // (Pedagogical explanation at http://aether.lbl.gov/www/classes/p139/homework/eight.pdf.
                // See "The Metric for a Uniformly Accelerating System.")
                TotalTimeWorld += DeltaTimeWorld;

                /*****************************
                 * PART 3 OF ALGORITHM
                 * FIND THE ROTATION MATRIX
                 * AND CHANGE THE PLAYERS VELOCITY
                 * BY THIS ROTATION MATRIX
                 * ***************************/


                //Find the turn angle
                //Steering constant angular velocity in the player frame
                //Rotate around the y-axis

                playerTransform.rotation = Quaternion.AngleAxis(playerRotation.y, Vector3.up) * Quaternion.AngleAxis(playerRotation.x, Vector3.right);
                // World rotation is opposite of player world rotation
                WorldRotation = CreateFromQuaternion(Quaternion.Inverse(playerTransform.rotation));

                //Add up our rotation so that we know where the character (NOT CAMERA) should be facing 
                playerRotation += deltaRotation;

                cameraForward = playerTransform.forward;
                deltaCameraAngle = Vector3.SignedAngle(oldCameraForward, cameraForward, playerTransform.up);
                if (deltaCameraAngle == 180)
                {
                    deltaCameraAngle = 0;
                }
                oldCameraForward = cameraForward;
            }

            _isInitDone = true;
        }

        protected void FixedUpdate()
        {
            Rigidbody playerRB = GameObject.FindGameObjectWithTag(Tags.playerRigidbody).GetComponent<Rigidbody>();

            if (!isMovementFrozen && (SpeedOfLight > 0))
            {
                if (isPlayerComoving)
                {
                    // Assume local player coordinates are comoving
                    Comovement cm = conformalMap.ComoveOptical(FixedDeltaTimePlayer, playerTransform.position, Quaternion.identity);
                    float test = cm.piw.sqrMagnitude;
                    if (!float.IsNaN(test) && !float.IsInfinity(test))
                    {
                        playerTransform.rotation = cm.riw * playerTransform.rotation;
                        playerTransform.position = cm.piw;
                    }
                }

                Vector3 pVel = -PlayerVelocityVector;
                playerRB.linearVelocity = pVel / SqrtOneMinusVSquaredCWDividedByCSquared;
                pVel = playerRB.linearVelocity;
                if (!IsPlayerFalling && (-pVel .y <= Physics.bounceThreshold)) {
                    Vector3 pVelPerp = new Vector3(pVel.x, 0, pVel.z);
                    playerRB.linearVelocity = pVel.AddVelocity(new Vector3(0, -pVel.y * pVelPerp.Gamma(), 0));
                }
            } else
            {
                playerRB.linearVelocity = Vector3.zero;
            }
        }
        #region Matrix/Quat math
        //They are functions that XNA had but Unity doesn't, so I had to make them myself

        //This function takes in a quaternion and creates a rotation matrix from it
        public Matrix4x4 CreateFromQuaternion(Quaternion q)
        {
            float w = q.w;
            float x = q.x;
            float y = q.y;
            float z = q.z;

            float wSqrd = w * w;
            float xSqrd = x * x;
            float ySqrd = y * y;
            float zSqrd = z * z;

            Matrix4x4 matrix = new Matrix4x4();
            matrix.SetColumn(0, new Vector4(wSqrd + xSqrd - ySqrd - zSqrd, 2 * x * y + 2 * w * z, 2 * x * z - 2 * w * y, 0));
            matrix.SetColumn(1, new Vector4(2 * x * y - 2 * w * z, wSqrd - xSqrd + ySqrd - zSqrd, 2 * y * z - 2 * w * x, 0));
            matrix.SetColumn(2, new Vector4(2 * x * z + 2 * w * y, 2 * y * z + 2 * w * x, wSqrd - xSqrd - ySqrd + zSqrd, 0));
            matrix.SetColumn(3, new Vector4(0, 0, 0, 1));

            return matrix;
        }
        #endregion
    }
}

namespace OpenRelativity
{
    public class PlayerController : RelativisticBehavior
    {
        // Consts
        protected const int INIT_FRAME_WAIT = 5;

        #region Public Parameters
        public float dragConstant = 0.75f;
        public float controllerAcceleration = 24;
        public bool useGravity = false;
        // If using comoveViaAcceleration, turn off isPlayerComoving in GameState.
        public bool comoveViaAcceleration = false;
        public float controllerBoost = 6000;
        // Affect our rotation speed
        public float rotSpeed;
        // For now, you can change this how you like.
        public float mouseSensitivity;
        // Keep track of the camera transform
        public Transform camTransform;
        #endregion

        //Needed to tell whether we are in free fall
        protected bool isFalling
        {
            get
            {
                return state.IsPlayerFalling;
            }

            set
            {
                state.IsPlayerFalling = value;
            }
        }
        public List<Collider> collidersBelow { get; protected set; }
        //Just turn this negative when they press the Y button for inversion.
        protected int inverted;
        //What is our current target for the speed of light?
        public int speedOfLightTarget { get; set; }
        //What is each step we take to reach that target?
        private float speedOfLightStep;
        //So we can use getAxis as keyHit function
        public bool invertKeyDown { get; set; }
        //Keep track of total frames passed
        protected int frames;
        protected Rigidbody myRigidbody;

        // Based on Strano 2019, (preprint).
        // (I will always implement potentially "cranky" features so you can toggle them off, but I might as well.)
        public bool isMonopoleAccel = false;
        public float monopoleAccelerationSoften = 0;
        // The composite scalar monopole graviton gas is described by statistical mechanics and heat flow equations
        public float gravitonEmissivity = 0.1f;
        // By default, 12g per baryon mole would be carbon-12, and this controls the total baryons estimated in the object
        public float fundamentalAverageMolarMass = 0.012f;
        public float initialAverageMolarMass = 0.012f;
        public float currentAverageMolarMass
        {
            get
            {
                if (!myRigidbody)
                {
                    return 0;
                }

                return myRigidbody.mass * SRelativityUtil.avogadroNumber / baryonCount;
            }
            protected set
            {
                if (myRigidbody)
                {
                    myRigidbody.mass = value * baryonCount / SRelativityUtil.avogadroNumber;
                }
            }
        }
        public Vector3 leviCivitaDevAccel = Vector3.zero;

        public float baryonCount { get; set; }

        public float monopoleTemperature
        {
            get
            {
                if (!myRigidbody)
                {
                    return 0;
                }

                // Per Strano 2019, due to the interaction with the thermal graviton gas radiated by the Rindler horizon,
                // there is also a change in mass. However, the monopole waves responsible for this are seen from a first-person perspective,
                // (i.e. as due to "player" acceleration).

                // If a gravitating body this RO is attracted to is already excited above the rest mass vacuum,
                // (which seems to imply the Higgs field vacuum)
                // then it will spontaneously emit this excitation, with a coupling constant proportional to the
                // gravitational constant "G" times (baryon) constituent particle rest mass.

                double nuclearMass = myRigidbody.mass / baryonCount;
                double fundamentalNuclearMass = fundamentalAverageMolarMass / SRelativityUtil.avogadroNumber;
                double excitationEnergy = (nuclearMass - fundamentalNuclearMass) * state.SpeedOfLightSqrd;
                double temperature = 2 * excitationEnergy / state.boltzmannConstant;

                return (float)temperature;

                //... But just turn "doDegradeAccel" off, if you don't want this effect for any reason.
            }

            set
            {
                if (!myRigidbody)
                {
                    return;
                }

                double fundamentalNuclearMass = fundamentalAverageMolarMass / SRelativityUtil.avogadroNumber;
                double excitationEnergy = value * state.boltzmannConstant / 2;
                double nuclearMass = excitationEnergy / state.SpeedOfLightSqrd + fundamentalNuclearMass;

                myRigidbody.mass = (float)(nuclearMass * baryonCount);
            }
        }

        //Keep track of our own Mesh Filter
        private MeshFilter meshFilter;

        virtual protected void Start()
        {
            collidersBelow = new List<Collider>();

            //same for RigidBody
            myRigidbody = state.playerTransform.GetComponent<Rigidbody>();
            //Assume we are in free fall
            isFalling = true;
            //If we have gravity, this factors into transforming to optical space.
            if (useGravity) state.HasWorldGravity = true;

            //Lock and hide cursor
            Cursor.lockState = CursorLockMode.Locked;
            //Cursor.visible = false;
            //Set the speed of light to the starting speed of light in GameState
            speedOfLightTarget = (int)state.SpeedOfLight;

            string[] names = Input.GetJoystickNames();
            // Inverted, at first
            inverted = (names.Length > 0) ? -1 : 1;
            invertKeyDown = false;

            frames = 0;

            meshFilter = transform.GetComponent<MeshFilter>();

            if (myRigidbody != null)
            {
                baryonCount = myRigidbody.mass * SRelativityUtil.avogadroNumber / currentAverageMolarMass;
            }
        }
        //Again, use LateUpdate to solve some collision issues.
        virtual protected void LateUpdate()
        {
            if (state.isMovementFrozen)
            {
                return;
            }

            Collider myColl = GetComponent<Collider>();
            Vector3 extents = myColl.bounds.extents;
            //We assume that the world "down" direction is the direction of gravity.
            Vector3 playerPos = state.playerTransform.position;
            Ray rayDown = new Ray(playerPos + extents.y * Vector3.down / 2, Vector3.down);
            RaycastHit hitInfo;
            // TODO: Layer mask
            isFalling = !Physics.Raycast(rayDown, out hitInfo, extents.y / 2);

            if (!isFalling)
            {
                if (myRigidbody.linearVelocity.y < 0)
                {
                    myRigidbody.linearVelocity = new Vector3(myRigidbody.linearVelocity.x, 0, myRigidbody.linearVelocity.z);
                }
            }

            //If we're not paused, update speed and rotation using player input.
            state.deltaRotation = Vector3.zero;

            //If they press the Y button, invert all Y axes
            if (Input.GetAxis("Invert Button") > 0 && !invertKeyDown)
            {
                inverted *= -1;
                invertKeyDown = true;
            }
            //And if they released it, set the invertkeydown to false.
            else if (!(Input.GetAxis("Invert Button") > 0))
            {
                invertKeyDown = false;
            }

            #region ControlScheme

            //PLAYER MOVEMENT

            //If we press W, move forward, if S, backwards.
            //A adds speed to the left, D to the right. We're using FPS style controls
            //Here's hoping they work.

            //The acceleration relation is defined by the following equation
            //vNew = (v+uParallel+ (uPerpendicular/gamma))/(1+(v*u)/c^2)

            //Okay, so Gerd found a good equation that doesn't break when your velocity is zero, BUT your velocity has to be in the x direction.
            //So, we're gonna reuse some code from our relativisticObject component, and rotate everything to be at the X axis.

            //Cache our velocity
            Vector3 playerVelocityVector = state.PlayerVelocityVector;

            Vector3 totalAccel = Vector3.zero;

            float temp;
            // Movement due to forward/back input
            totalAccel += new Vector3(0, 0, (temp = inverted * -Input.GetAxis("Vertical")) * controllerAcceleration);
            if (temp != 0)
            {
                state.keyHit = true;
            }
            // Movement due to left/right input
            totalAccel += new Vector3((temp = -Input.GetAxis("Horizontal")) * controllerAcceleration, 0, 0);
            if (temp != 0)
            {
                state.keyHit = true;
            }

            //Turn our camera rotation into a Quaternion. This allows us to make where we're pointing the direction of our added velocity.
            //If you want to constrain the player to just x/z movement, with no Y direction movement, comment out the next two lines
            //and uncomment the line below that is marked
            Quaternion cameraRotation = Quaternion.LookRotation(camTransform.forward, camTransform.up);

            //UNCOMMENT THIS LINE if you would like to constrain the player to just x/z movement.
            //Quaternion cameraRotation = Quaternion.AngleAxis(camTransform.eulerAngles.y, Vector3.up);

            //And rotate our added velocity by camera angle
            totalAccel = cameraRotation * totalAccel;

            //AUTO SLOW DOWN CODE BLOCK

            //Add a fluid drag force (as for air)
            totalAccel -= dragConstant * playerVelocityVector.sqrMagnitude * playerVelocityVector.normalized;

            Vector3 quasiWorldAccel = totalAccel;

            if (comoveViaAcceleration)
            {
                // Unlike RelativisticObject instances, this is optionally how the player "comoves."
                // If using comoveViaAcceleration, turn off isPlayerComoving in GameState.
                quasiWorldAccel -= state.conformalMap.GetRindlerAcceleration(state.playerTransform.position);
            }

            if (isFalling)
            {
                if (useGravity)
                {
                    quasiWorldAccel -= Physics.gravity;
                }
            }
            else
            {
                if (quasiWorldAccel.y < 0)
                {
                    quasiWorldAccel.y = 0;
                }

                if (totalAccel.y < 0)
                {
                    totalAccel.y = 0;
                }

                if (useGravity)
                {
                    totalAccel -= Physics.gravity;
                    quasiWorldAccel = new Vector3(quasiWorldAccel.x, 0, quasiWorldAccel.z);
                }

                totalAccel -= state.conformalMap.GetRindlerAcceleration(state.playerTransform.position);
            }

            if (isMonopoleAccel)
            {
                // Per Strano 2019, acceleration "nudges" the preferred accelerated rest frame.
                // (Relativity privileges no "inertial" frame, but there is intrinsic observable difference between "accelerated frames.")
                // (The author speculates, this accelerated frame "nudge" might be equivalent to the 3-vector potential of the Higgs field.
                // The scalar potential can excite the "fundamental" rest mass. The independence of the rest mass from gravitational acceleration
                // has been known since Galileo.)

                // If a gravitating body this RO is attracted to is already excited above the rest mass vacuum,
                // (which seems to imply the Higgs field vacuum)
                // then it will spontaneously emit this excitation, with a coupling constant proportional to the
                // gravitational constant "G" times (baryon) constituent particle rest mass.
                // (For video game purposes, there's maybe no easy way to precisely model the mass flow, so just control it with an editor variable.)

                quasiWorldAccel += leviCivitaDevAccel;

                float softenFactor = 1 + monopoleAccelerationSoften;
                float tempSoftenFactor = Mathf.Pow(softenFactor, 1.0f / 4);

                monopoleTemperature /= tempSoftenFactor;
                float origBackgroundTemp = state.gravityBackgroundPlanckTemperature;
                state.gravityBackgroundPlanckTemperature /= tempSoftenFactor;

                EvaporateMonopole(softenFactor * Time.deltaTime, totalAccel / softenFactor);

                state.gravityBackgroundPlanckTemperature = origBackgroundTemp;
                monopoleTemperature *= tempSoftenFactor;
            }

            //3-acceleration acts as classically on the rapidity, rather than velocity.
            Vector3 totalVel = playerVelocityVector.AddVelocity((quasiWorldAccel * Time.deltaTime).RapidityToVelocity());
            Vector3 projVOnG = Vector3.Project(totalVel, Physics.gravity);
            if (useGravity && !isFalling && ((projVOnG - Physics.gravity).sqrMagnitude <= SRelativityUtil.FLT_EPSILON))
            {
                totalVel = totalVel.AddVelocity(projVOnG * totalVel.Gamma());
                totalVel = new Vector3(totalVel.x, 0, totalVel.z);
            }

            float tvMag = totalVel.magnitude;

            if (tvMag >= state.maxPlayerSpeed - .01f)
            {
                float gamma = totalVel.Gamma();
                Vector3 diff = totalVel.normalized * (state.maxPlayerSpeed - .01f) - totalVel;
                totalVel += diff;
                totalAccel += diff * gamma;
            }

            state.PlayerVelocityVector = totalVel;
            state.PlayerAccelerationVector = totalAccel;

            //CHANGE the speed of light

            //Get our input axis (DEFAULT N, M) value to determine how much to change the speed of light
            int temp2 = (int)(Input.GetAxis("Speed of Light"));
            //If it's too low, don't subtract from the speed of light, and reset the speed of light
            if (temp2 < 0 && speedOfLightTarget <= state.MaxSpeed)
            {
                temp2 = 0;
                speedOfLightTarget = (int)state.MaxSpeed;
            }
            if (temp2 != 0)
            {
                speedOfLightTarget += temp2;

                speedOfLightStep = Mathf.Abs((state.SpeedOfLight - speedOfLightTarget) / 20);
            }
            //Now, if we're not at our target, move towards the target speed that we're hoping for
            if (state.SpeedOfLight < speedOfLightTarget * .995f)
            {
                //Then we change the speed of light, so that we get a smooth change from one speed of light to the next.
                state.SpeedOfLight += speedOfLightStep;
            }
            else if (state.SpeedOfLight > speedOfLightTarget * 1.005f)
            {
                //See above
                state.SpeedOfLight -= speedOfLightStep;
            }
            //If we're within a +-.05 distance of our target, just set it to be our target.
            else if (state.SpeedOfLight != speedOfLightTarget)
            {
                state.SpeedOfLight = speedOfLightTarget;
            }

            //MOUSE CONTROLS
            //Current position of the mouse
            //Difference between last frame's mouse position
            //X axis position change
            float positionChangeX = -Input.GetAxis("Mouse X");

            //Y axis position change
            float positionChangeY = inverted * -Input.GetAxis("Mouse Y");

            //Use these to determine camera rotation, that is, to look around the world without changing direction of motion
            //These two are for X axis rotation and Y axis rotation, respectively
            float viewRotX, viewRotY;
            if (Mathf.Abs(positionChangeX) <= 1 && Mathf.Abs(positionChangeY) <= 1)
            {
                //Take the position changes and translate them into an amount of rotation
                viewRotX = -positionChangeX * Time.deltaTime * rotSpeed * mouseSensitivity * controllerBoost;
                viewRotY = positionChangeY * Time.deltaTime * rotSpeed * mouseSensitivity * controllerBoost;
            }
            else
            {
                //Take the position changes and translate them into an amount of rotation
                viewRotX = -positionChangeX * Time.deltaTime * rotSpeed * mouseSensitivity;
                viewRotY = positionChangeY * Time.deltaTime * rotSpeed * mouseSensitivity;
            }
            //Perform Rotation on the camera, so that we can look in places that aren't the direction of movement
            //Wait some frames on start up, otherwise we spin during the intialization when we can't see yet
            if (frames > INIT_FRAME_WAIT)
            {
                camTransform.Rotate(new Vector3(0, viewRotX, 0), Space.World);
                if ((camTransform.eulerAngles.x + viewRotY < 90 && camTransform.eulerAngles.x + viewRotY > 90 - 180) || (camTransform.eulerAngles.x + viewRotY > 270 && camTransform.eulerAngles.x + viewRotY < 270 + 180))
                {
                    camTransform.Rotate(new Vector3(viewRotY, 0, 0));
                }
            }
            else
            {
                //keep track of our frames
                ++frames;
            }

            //If we have a speed of light less than max speed, fix it.
            //This should never happen
            if (state.SpeedOfLight < state.MaxSpeed)
            {
                state.SpeedOfLight = state.MaxSpeed;
            }


            #endregion

            //Send current speed of light to the shader
            Shader.SetGlobalFloat("_spdOfLight", state.SpeedOfLight);

            if (Camera.main)
            {
                Shader.SetGlobalFloat("xyr", Camera.main.pixelWidth / Camera.main.pixelHeight);
                Shader.SetGlobalFloat("xs", Mathf.Tan(Mathf.Deg2Rad * Camera.main.fieldOfView / 2f));

                //Don't cull because at high speeds, things come into view that are not visible normally
                //This is due to the lorenz transformation, which is pretty cool but means that basic culling will not work.
                Camera.main.layerCullSpherical = true;
                Camera.main.useOcclusionCulling = false;
            }
        }

        protected void EvaporateMonopole(float deltaTime, Vector3 myAccel)
        {
            // If the RelativisticObject is at rest on the ground, according to Strano 2019, (not yet peer reviewed,)
            // it loses surface acceleration, (not weight force, directly,) the longer it stays in this configuration.
            // The Rindler horizon evaporates as would Schwarzschild, for event horizon surface acceleration equivalent
            // between the Rindler and Schwarzschild metrics. Further, Hawking(-Unruh, et al.) acceleration might have
            // the same effect.

            // The Rindler horizon evaporates as a Schwarzschild event horizon with the same surface gravity, according to Strano.
            // We add any background radiation power.
            double alpha = myAccel.magnitude;
            bool isNonZeroTemp = alpha > SRelativityUtil.FLT_EPSILON;

            double r = double.PositiveInfinity;
            // If alpha is in equilibrium with the background temperature, there is no evaporation.
            if (isNonZeroTemp)
            {
                // Surface acceleration at event horizon:
                r = state.SpeedOfLightSqrd / (2 * alpha);
                r = SRelativityUtil.EffectiveRaditiativeRadius((float)r, state.gravityBackgroundPlanckTemperature);
            }

            if (r < state.planckLength)
            {
                // For minimum area calculation, below.
                r = state.planckLength;
            }

            if (!double.IsInfinity(r) && !double.IsNaN(r))
            {
                isNonZeroTemp = true;
                r += SRelativityUtil.SchwarzschildRadiusDecay(deltaTime, r);
                if (r <= SRelativityUtil.FLT_EPSILON)
                {
                    leviCivitaDevAccel += myAccel;
                }
                else
                {
                    double alphaF = state.SpeedOfLightSqrd / (2 * r);
                    leviCivitaDevAccel += (float)(alpha - alphaF) * myAccel.normalized;
                }
            }

            if (myRigidbody != null)
            {
                double myTemperature = monopoleTemperature;
                double surfaceArea;
                if (meshFilter == null)
                {
                    Vector3 lwh = transform.localScale;
                    surfaceArea = 2 * (lwh.x * lwh.y + lwh.x * lwh.z + lwh.y * lwh.z);
                }
                else
                {
                    surfaceArea = meshFilter.sharedMesh.SurfaceArea();
                }
                // This is the ambient temperature, including contribution from comoving accelerated rest temperature.
                double ambientTemperature = isNonZeroTemp ? SRelativityUtil.SchwarzRadiusToPlanckScaleTemp(r) : state.gravityBackgroundPlanckTemperature;
                double dm = (gravitonEmissivity * surfaceArea * SRelativityUtil.sigmaPlanck * (Math.Pow(myTemperature, 4) - Math.Pow(ambientTemperature, 4))) / state.planckArea;

                // Momentum is conserved. (Energy changes.)
                Vector3 momentum = myRigidbody.mass * state.PlayerVelocityVector;

                double camm = (myRigidbody.mass - dm) * SRelativityUtil.avogadroNumber / baryonCount;

                if ((myTemperature > 0) && (camm < fundamentalAverageMolarMass))
                {
                    currentAverageMolarMass = fundamentalAverageMolarMass;
                }
                else if (camm <= 0)
                {
                    myRigidbody.mass = 0;
                }
                else
                {
                    myRigidbody.mass -= (float)dm; 
                }

                if (myRigidbody.mass > SRelativityUtil.FLT_EPSILON)
                {
                    state.PlayerVelocityVector = momentum / myRigidbody.mass;
                }
            }
        }

        protected void OnTriggerEnter(Collider collider)
        {
            OnTrigger(collider);
        }

        protected void OnTriggerStay(Collider collider)
        {
            OnTrigger(collider);
        }

        //Note that this method assumes that walls are locked at right angles to the world coordinate axes.
        private void OnTrigger(Collider collider)
        {
            if (collider.isTrigger)
            {
                return;
            }

            Rigidbody otherRB = collider.GetComponent<Rigidbody>();

            if (otherRB != null && !otherRB.isKinematic)
            {
                return;
            }

            // Vector3 origPlayerVel = state.PlayerVelocityVector;

            Collider myColl = GetComponent<Collider>();
            Vector3 extents = myColl.bounds.extents;
            //We assume that the world "down" direction is the direction of gravity.
            Vector3 playerPos = state.playerTransform.position;
            Ray rayDown = new Ray(playerPos + extents.y * Vector3.down / 2, Vector3.down);
            Ray rayUp = new Ray(playerPos + extents.y * Vector3.down, Vector3.up);
            Ray rayLeft = new Ray(playerPos + extents.x * Vector3.right, Vector3.left);
            Ray rayRight = new Ray(playerPos + extents.x * Vector3.left, Vector3.right);
            Ray rayForward = new Ray(playerPos + extents.z * Vector3.back, Vector3.forward);
            Ray rayBack = new Ray(playerPos + extents.z * Vector3.forward, Vector3.back);
            RaycastHit hitInfo;
            float dist;
            if (collider.Raycast(rayDown, out hitInfo, extents.y / 2))
            {
                if (frames > INIT_FRAME_WAIT)
                {
                    Vector3 pVel = state.PlayerVelocityVector;
                    if (pVel.y > 0)
                    {
                        Vector3 pVelPerp = new Vector3(pVel.x, 0, pVel.z);
                        state.PlayerVelocityVector = state.PlayerVelocityVector.AddVelocity(new Vector3(0, -pVel.y * pVelPerp.Gamma(), 0));
                        Vector3 totalVel = state.PlayerVelocityVector;
                        state.PlayerVelocityVector = new Vector3(totalVel.x, 0, totalVel.z);
                        Rigidbody myRB = transform.parent.GetComponent<Rigidbody>();
                        myRB.linearVelocity = new Vector3(myRB.linearVelocity.x, 0, myRB.linearVelocity.z);
                    }

                    dist = extents.y / 2 - hitInfo.distance;
                    if (dist > 0.05f)
                    {
                        Vector3 pos = state.playerTransform.position;
                        state.playerTransform.position = new Vector3(pos.x, pos.y + dist, pos.z);
                    }
                }
            }

            Vector3 direction = Vector3.zero;
            if (collider.Raycast(rayForward, out hitInfo, 2 * extents.z))
            {
                direction = Vector3.forward;
            }
            else if (collider.Raycast(rayBack, out hitInfo, 2 * extents.z))
            {
                direction = Vector3.back;
            }
            else if (collider.Raycast(rayLeft, out hitInfo, 2 * extents.x))
            {
                direction = Vector3.left;
            }
            else if (collider.Raycast(rayRight, out hitInfo, 2 * extents.x))
            {
                direction = Vector3.right;
            }
            else if (collider.Raycast(rayUp, out hitInfo, 2 * extents.y))
            {
                direction = Vector3.up;
            }

            if (direction != Vector3.zero)
            {
                Vector3 pVel = state.PlayerVelocityVector;
                if (Vector3.Dot(pVel, direction) < 0)
                {
                    //Decompose velocity in parallel and perpendicular components:
                    Vector3 myParraVel = Vector3.Project(pVel, direction) * 2;
                    //Vector3 myPerpVel = Vector3.Cross(direction, Vector3.Cross(direction, pVel));
                    //Relativistically cancel the downward velocity:
                    state.PlayerVelocityVector = state.PlayerVelocityVector - myParraVel;
                }
            }

            // Vector3 accel = (state.PlayerVelocityVector - origPlayerVel) / state.FixedDeltaTimePlayer;
            // EvaporateMonopole(state.FixedDeltaTimePlayer, accel);
        }
    }
}

namespace OpenRelativity.Objects
{
    [ExecuteInEditMode]
    public class RelativisticObject : RelativisticBehavior
    {
        #region Public Settings
        public bool isLightMapStatic = false;
        public bool useGravity = false;
        // Use this instead of relativistic parent
        public bool isParent = false;
        // Combine colliders under us in the hierarchy
        public bool isCombinedColliderParent = false;
        // Use this if not using an explicitly relativistic shader
        public bool isNonrelativisticShader = false;
        // We set the Rigidbody "drag" parameter in this object.
        public float unityDrag = 0;
        // We also set the Rigidbody "angularDrag" parameter in this object.
        public float unityAngularDrag = 0;
        // Comove via ConformalMap acceleration, or ComoveOptical?
        public bool comoveViaAcceleration = false;
        // The composite scalar monopole graviton gas is described by statistical mechanics and heat flow equations
        public float gravitonEmissivity = 0.1f;
        // By default, 12g per baryon mole would be carbon-12, and this controls the total baryons estimated in the object
        public float fundamentalAverageMolarMass = 0.012f;
        public float initialAverageMolarMass = 0.012f;
        public float currentAverageMolarMass
        {
            get
            {
                if (myRigidbody)
                {
                    return myRigidbody.mass * SRelativityUtil.avogadroNumber / baryonCount;
                }

                return 0;
            }
            protected set
            {
                if (myRigidbody)
                {
                    myRigidbody.mass = value * baryonCount / SRelativityUtil.avogadroNumber;
                }
            }
        }

        // Set with Rigidbody isKinematic flag instead
        public bool isKinematic
        {
            get
            {
                if (myRigidbody)
                {
                    return myRigidbody.isKinematic;
                }

                return false;
            }

            set
            {
                if (myRigidbody)
                {
                    myRigidbody.isKinematic = value;
                }
            }
        }
        #endregion

        protected bool isPhysicsUpdateFrame;

        protected float updateViwTimeFactor;
        protected float updatePlayerViwTimeFactor;
        protected float updateTisw;
        protected Matrix4x4 updateMetric;
        protected Vector4 updateWorld4Acceleration;

        #region Local Time
        //Acceleration desyncronizes our clock from the world clock:
        public float localTimeOffset { get; private set; }
        private float oldLocalTimeUpdate;
        public float localDeltaTime
        {
            get
            {
                return GetLocalTime() - oldLocalTimeUpdate;
            }
        }
        public float localFixedDeltaTime
        {
            get
            {
                return GetLocalTime() - oldLocalTimeFixedUpdate;
            }
        }
        private float oldLocalTimeFixedUpdate;
        public float GetLocalTime()
        {
            return state.TotalTimeWorld + localTimeOffset;
        }
        public void ResetLocalTime()
        {
            localTimeOffset = 0;
        }
        public float GetTisw()
        {
            if (isPhysicsCacheValid)
            {
                return updateTisw;
            }

            return piw.GetTisw(viw, GetComoving4Acceleration());
        }
        public float GetVisualTime()
        {
            return GetLocalTime() + GetTisw();
        }
        #endregion

        #region 4-vector relativity
        // This is the metric tensor in an accelerated frame in special relativity.
        // Special relativity assumes a flat metric, (the "Minkowski metric").
        // In general relativity, the underlying metric could be curved, according to the Einstein field equations.
        // The (flat) metric appears to change due to proper acceleration from the player's/camera's point of view, since acceleration is not physically relative like velocity.
        // (Physically, proper acceleration could be detected by a force on the observer in the opposite direction from the acceleration,
        // like being pushed back into the seat of an accelerating car. When we stand still on the surface of earth, we feel our weight as the ground exerts a normal force "upward,"
        // which is equivalent to an acceleration in the opposite direction from the ostensible Newtonian gravity field, similar to the car.
        // "Einstein equivalence principle" says that, over small enough regions, we can't tell the difference between
        // a uniform acceleration and a gravitational field, that the two are physically equivalent over small enough regions of space.
        // In free-fall, gravitational fields disappear. Hence, when the player is in free-fall, their acceleration is considered to be zero,
        // while it is considered to be "upwards" when they are at rest under the effects of gravity, so they don't fall through the surface they're feeling pushed into.)
        // The apparent deformation of the Minkowski metric also depends on an object's distance from the player, so it is calculated by and for the object itself.

        public Matrix4x4 GetMetric()
        {
            if (isPhysicsCacheValid)
            {
                return updateMetric;
            }

            Matrix4x4 metric = SRelativityUtil.GetRindlerMetric(piw);
            Matrix4x4 invVpcLorentz = state.PlayerLorentzMatrix.inverse;
            metric = invVpcLorentz.transpose * metric * invVpcLorentz;

            Matrix4x4 intrinsicMetric = state.conformalMap.GetMetric(piw);
            return intrinsicMetric * metric * intrinsicMetric.inverse;
        }

        public Vector4 Get4Velocity()
        {
            return viw.ToMinkowski4Viw();
        }

        public Vector4 GetComoving4Acceleration()
        {
            if (isPhysicsCacheValid)
            {
                return updateWorld4Acceleration;
            }

            return comovingAccel.ProperToWorldAccel(viw, GetTimeFactor());
        }

        public Vector4 GetWorld4Acceleration()
        {
            return aiw.ProperToWorldAccel(viw, GetTimeFactor());
        }

        // This is the factor commonly referred to as "gamma," for length contraction and time dilation,
        // only also with consideration for a gravitationally curved background, such as due to Rindler coordinates.
        // (Rindler coordinates are actually Minkowski flat, but the same principle applies.)
        public float GetTimeFactor()
        {
            if (isPhysicsCacheValid)
            {
                return updateViwTimeFactor;
            }

            // However, sometimes we want a different velocity, at this space-time point,
            // such as this RO's own velocity.
            return viw.InverseGamma(GetMetric());
        }
        #endregion

        #region Rigid body physics
        private bool wasKinematic;
        private CollisionDetectionMode collisionDetectionMode;
        private PhysicsMaterial[] origPhysicMaterials;

        private Vector3 oldVelocity;
        private float lastFixedUpdateDeltaTime;

        public float baryonCount { get; set; }

        public float mass
        {
            get
            {
                if (myRigidbody)
                {
                    return myRigidbody.mass;
                }

                return 0;
            }

            set
            {
                if (myRigidbody)
                {
                    myRigidbody.mass = value;
                }
            }
        }

        public void AddForce(Vector3 force, ForceMode mode = ForceMode.Force)
        {
            if (!myRigidbody) {
                if (mode == ForceMode.Impulse)
                {
                    return;
                }
                if (mode == ForceMode.Force)
                {
                    return;
                }
            }

            switch (mode)
            {
                case ForceMode.Impulse:
                    peculiarVelocity += force / mass;
                    break;
                case ForceMode.VelocityChange:
                    peculiarVelocity += force;
                    break;
                case ForceMode.Force:
                    nonGravAccel += force / mass;
                    break;
                case ForceMode.Acceleration:
                    nonGravAccel += force;
                    break;
            }
        }

        //Store world position, mostly for a nonrelativistic shader:
        protected Vector3 _piw;
        public Vector3 piw {
            get
            {
                return _piw;
            }
            set
            {
                if ((value - _piw).sqrMagnitude <= SRelativityUtil.FLT_EPSILON) {
                    return;
                }

                _piw = value;
                UpdatePhysicsCaches();
            }
        }

        public Vector3 opticalPiw {
            get
            {
                return piw.WorldToOptical(peculiarVelocity, GetComoving4Acceleration());
            }
            set
            {
                piw = value.OpticalToWorld(peculiarVelocity, GetComoving4Acceleration());
            }
        }

        public void ResetPiw()
        {
            piw = isNonrelativisticShader ? transform.position.OpticalToWorld(peculiarVelocity, GetComoving4Acceleration()) : transform.position;
        }
        //Store rotation quaternion
        public Quaternion riw { get; set; }

        public Vector3 _peculiarVelocity = Vector3.zero;
        public Vector3 peculiarVelocity
        {
            get
            {
                return _peculiarVelocity;
            }

            set
            {
                // Skip this all, if the change is negligible.
                if ((value - _peculiarVelocity).sqrMagnitude <= SRelativityUtil.FLT_EPSILON)
                {
                    return;
                }

                UpdateMotion(value, nonGravAccel);
                UpdateRigidbodyVelocity();
            }
        }

        public Vector3 vff
        {
            get
            {
                return state.conformalMap ? state.conformalMap.GetFreeFallVelocity(piw) : Vector3.zero;
            }
        }

        public Vector3 viw
        {
            get
            {
                return vff.AddVelocity(peculiarVelocity);
            }

            set
            {
                peculiarVelocity = (-vff).AddVelocity(value);
            }
        }

        //Store this object's angular velocity here.
        public Vector3 _aviw;
        public Vector3 aviw
        {
            get
            {
                return _aviw;
            }
            set
            {
                if (!isKinematic)
                {
                    _aviw = value;
                    UpdateRigidbodyVelocity();
                }
            }
        }

        // This is the part of acceleration that can be set.
        public Vector3 _nonGravAccel;
        public Vector3 nonGravAccel
        {
            get
            {
                return _nonGravAccel;
            }

            set
            {
                // Skip this all, if the change is negligible.
                if (isKinematic || (value - _nonGravAccel).sqrMagnitude <= SRelativityUtil.FLT_EPSILON)
                {
                    return;
                }

                UpdateMotion(peculiarVelocity, value);
                UpdateRigidbodyVelocity();
            }
        }

        // This would be the object's properAccleration if at rest in "world coordinates."
        public Vector3 aiw
        {
            get
            {
                Vector3 accel = nonGravAccel + leviCivitaDevAccel;

                if (useGravity)
                {
                    accel += Physics.gravity;
                }

                accel += state.conformalMap.GetRindlerAcceleration(piw);

                return accel;
            }

            set
            {
                Vector3 accel = value;

                if (state.conformalMap)
                {
                    accel -= state.conformalMap.GetRindlerAcceleration(piw);
                }

                if (useGravity)
                {
                    accel -= Physics.gravity;
                }

                accel -= leviCivitaDevAccel;

                nonGravAccel = accel;
            }
        }

        // This would be the object's "proper acceleration" in comoving coordinates,
        // but we compose with Rindler metric for Physics.gravity, so that we can
        // keep book as the Physics.gravity vector effect being an intrinsic property.
        public Vector3 comovingAccel
        {
            get
            {
                if (myRigidbody && myRigidbody.IsSleeping())
                {
                    return Vector3.zero;
                }

                return nonGravAccel;
            }
            set
            {
                nonGravAccel = value;
            }
        }

        public float monopoleTemperature
        {
            get
            {
                if (!myRigidbody)
                {
                    return 0;
                }

                // Per Strano 2019, due to the interaction with the thermal graviton gas radiated by the Rindler horizon,
                // there is also a change in mass. However, the monopole waves responsible for this are seen from a first-person perspective,
                // (i.e. as due to "player" acceleration).

                // If a gravitating body this RO is attracted to is already excited above the rest mass vacuum,
                // (which seems to imply the Higgs field vacuum)
                // then it will spontaneously emit this excitation, with a coupling constant proportional to the
                // gravitational constant "G" times (baryon) constituent particle rest mass.

                double nuclearMass = mass / baryonCount;
                double fundamentalNuclearMass = fundamentalAverageMolarMass / SRelativityUtil.avogadroNumber;

                if (nuclearMass < fundamentalNuclearMass)
                {
                    // We can't support negative temperatures, yet, for realistic constants,
                    // (not that we'd necessarily want them).
                    return 0;
                }

                double excitationEnergy = (nuclearMass - fundamentalNuclearMass) * state.SpeedOfLightSqrd;
                double temperature = 2 * excitationEnergy / state.boltzmannConstant;

                return (float)temperature;

                //... But just turn "doDegradeAccel" off, if you don't want this effect for any reason.
            }

            set
            {
                if (!myRigidbody)
                {
                    return;
                }

                double fundamentalNuclearMass = fundamentalAverageMolarMass / SRelativityUtil.avogadroNumber;
                double excitationEnergy = value * state.boltzmannConstant / 2;
                if (excitationEnergy < 0)
                {
                    excitationEnergy = 0;
                }
                double nuclearMass = excitationEnergy / state.SpeedOfLightSqrd + fundamentalNuclearMass;

                mass = (float)(nuclearMass * baryonCount);
            }
        }

        public Vector3 leviCivitaDevAccel = Vector3.zero;

        public void UpdateMotion(Vector3 pvf, Vector3 af)
        {
            // Changing velocities lose continuity of position,
            // unless we transform the world position to optical position with the old velocity,
            // and inverse transform the optical position with the new the velocity.
            // (This keeps the optical position fixed.)

            Vector3 pvi = peculiarVelocity;
            Vector3 ai = comovingAccel;
            _nonGravAccel = af;
            _peculiarVelocity = pvf;

            float timeFac = GetTimeFactor();

            _piw = piw.WorldToOptical(pvi, ai.ProperToWorldAccel(pvi, timeFac)).OpticalToWorld(pvf, comovingAccel.ProperToWorldAccel(pvf, timeFac));

            if (isNonrelativisticShader)
            {
                UpdateContractorPosition();
            }
            else
            {
                transform.position = piw;
            }

            UpdatePhysicsCaches();
            UpdateShaderParams();
        }
        #endregion

        #region Nonrelativistic Shader/Collider
        //If the shader is not relativistic, we need to handle length contraction with a "contractor" transform.
        private Transform contractor;
        //Changing a transform's parent is expensive, but we can avoid it with this:
        public Vector3 _localScale;
        public Vector3 localScale
        {
            get
            {
                return _localScale;
            }
            set
            {
                transform.localScale = value;
                _localScale = value;
            }
        }
        //If we specifically have a mesh collider, we need to know to transform the verts of the mesh itself.
        private bool isMyColliderMesh;
        private bool isMyColliderVoxel;
        private bool isMyColliderGeneral {
            get {
                return !isMyColliderMesh && !isMyColliderVoxel;
            }
        }

        //If we have a collider to transform, we cache it here
        private Collider[] myColliders;
        private SphereCollider[] mySphereColliders;
        private BoxCollider[] myBoxColliders;
        private CapsuleCollider[] myCapsuleColliders;

        private Vector3[] colliderPiw { get; set; }
        public void MarkStaticColliderPos()
        {
            if (isMyColliderGeneral)
            {
                List<Vector3> sttcPosList = new List<Vector3>();

                for (int i = 0; i < mySphereColliders.Length; ++i)
                {
                    sttcPosList.Add(mySphereColliders[i].center);
                }

                for (int i = 0; i < myBoxColliders.Length; ++i)
                {
                    sttcPosList.Add(myBoxColliders[i].center);
                }

                for (int i = 0; i < myCapsuleColliders.Length; ++i)
                {
                    sttcPosList.Add(myCapsuleColliders[i].center);
                }

                colliderPiw = sttcPosList.ToArray();
            }
        }
        #endregion

        #region RelativisticObject properties and caching
        //Don't render if object has relativistic parent
        private bool hasParent = false;
        //Keep track of our own Mesh Filter
        private MeshFilter meshFilter;

        //When was this object created? use for moving objects
        private bool hasStarted;
        private float _StartTime = float.NegativeInfinity;
        public float StartTime { get { return _StartTime; } set { _StartTime = value; } }
        //When should we die? again, for moving objects
        private float _DeathTime = float.PositiveInfinity;
        public float DeathTime { get { return _DeathTime; } set { _DeathTime = value; } }

        //We save and reuse the transformed vert array to avoid garbage collection 
        private Vector3[] trnsfrmdMeshVerts;
        //We create a new collider mesh, so as not to interfere with primitives, and reuse it
        private Mesh trnsfrmdMesh;
        //If we have a Rigidbody, we cache it here
        private Rigidbody myRigidbody;
        //If we have a Renderer, we cache it, too.
        public Renderer myRenderer { get; set; }

        //We need to freeze any attached rigidbody if the world states is frozen 
        public bool wasFrozen { get; set; }

        // Based on Strano 2019, (preprint).
        // (I will always implement potentially "cranky" features so you can toggle them off, but I might as well.)
        public bool isMonopoleAccel = false;
        public float monopoleAccelerationSoften = 0;
        #endregion

        #region Collider transformation and update
        // We use an attached shader to transform the collider verts:
        public ComputeShader colliderShader;
        // If the object is light map static, we need a duplicate of its mesh
        public Mesh colliderShaderMesh;
        // We set global constants in a struct
        private ShaderParams colliderShaderParams;
        // Mesh collider params
        private ComputeBuffer paramsBuffer;
        // Mesh collider vertices
        private ComputeBuffer vertBuffer;

        private void UpdateMeshCollider(MeshCollider transformCollider)
        {
            if (colliderShaderMesh == null || colliderShaderMesh.vertexCount == 0)
            {
                return;
            }

            if (paramsBuffer == null)
            {
                paramsBuffer = new ComputeBuffer(1, System.Runtime.InteropServices.Marshal.SizeOf(colliderShaderParams));
            }

            if (vertBuffer == null)
            {
                vertBuffer = new ComputeBuffer(colliderShaderMesh.vertexCount, System.Runtime.InteropServices.Marshal.SizeOf(new Vector3()));
            }

            //Freeze the physics if the global state is frozen.
            if (state.isMovementFrozen)
            {
                if (!wasFrozen)
                {
                    //Read the state of the rigidbody and shut it off, once.
                    wasFrozen = true;
                    if (!myRigidbody) {
                        wasKinematic = true;
                        collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
                    }
                    else      
                    {
                        wasKinematic = myRigidbody.isKinematic;
                        collisionDetectionMode = myRigidbody.collisionDetectionMode;
                        myRigidbody.collisionDetectionMode = CollisionDetectionMode.ContinuousSpeculative;
                        myRigidbody.isKinematic = true;
                    }
                }

                return;
            }
            
            if (wasFrozen)
            {
                //Restore the state of the rigidbody, once.
                wasFrozen = false;
                if (myRigidbody)
                {
                    myRigidbody.isKinematic = wasKinematic;
                    myRigidbody.collisionDetectionMode = collisionDetectionMode;
                }
            }

            //Set remaining global parameters:
            colliderShaderParams.ltwMatrix = transform.localToWorldMatrix;
            colliderShaderParams.wtlMatrix = transform.worldToLocalMatrix;
            colliderShaderParams.vpc = -state.PlayerVelocityVector / state.SpeedOfLight;
            colliderShaderParams.pap = state.PlayerAccelerationVector;
            colliderShaderParams.avp = state.PlayerAngularVelocityVector;
            colliderShaderParams.playerOffset = state.playerTransform.position;
            colliderShaderParams.spdOfLight = state.SpeedOfLight;
            colliderShaderParams.vpcLorentzMatrix = state.PlayerLorentzMatrix;
            colliderShaderParams.invVpcLorentzMatrix = state.PlayerLorentzMatrix.inverse;
            Matrix4x4 metric = state.conformalMap.GetMetric(piw);
            colliderShaderParams.intrinsicMetric = metric;
            colliderShaderParams.invIntrinsicMetric = metric.inverse;

            ShaderParams[] spa = new ShaderParams[1];
            spa[0] = colliderShaderParams;
            //Put verts in R/W buffer and dispatch:
            paramsBuffer.SetData(spa);
            vertBuffer.SetData(colliderShaderMesh.vertices);
            int kernel = colliderShader.FindKernel("CSMain");
            colliderShader.SetBuffer(kernel, "glblPrms", paramsBuffer);
            colliderShader.SetBuffer(kernel, "verts", vertBuffer);
            colliderShader.Dispatch(kernel, colliderShaderMesh.vertexCount, 1, 1);
            vertBuffer.GetData(trnsfrmdMeshVerts);

            //Change mesh:
            trnsfrmdMesh.vertices = trnsfrmdMeshVerts;
            trnsfrmdMesh.RecalculateBounds();
            trnsfrmdMesh.RecalculateNormals();
            trnsfrmdMesh.RecalculateTangents();
            transformCollider.sharedMesh = trnsfrmdMesh;

            if (myRigidbody)
            {
                myRigidbody.ResetCenterOfMass();
                myRigidbody.ResetInertiaTensor();
            }
        }

        public void UpdateColliders()
        {
            MeshCollider[] myMeshColliders = GetComponents<MeshCollider>();

            //Get the vertices of our mesh
            if (!colliderShaderMesh && meshFilter && meshFilter.sharedMesh.isReadable)
            {
                colliderShaderMesh = Instantiate(meshFilter.sharedMesh);
            }

            if (colliderShaderMesh)
            {
                trnsfrmdMesh = Instantiate(colliderShaderMesh);
                trnsfrmdMeshVerts = (Vector3[])trnsfrmdMesh.vertices.Clone();
                trnsfrmdMesh.MarkDynamic();

                if (!enabled || !gameObject.activeInHierarchy)
                {
                    UpdateMeshCollider(myMeshColliders[0]);
                }
            }

            if (GetComponent<ObjectBoxColliderDensity>())
            {
                isMyColliderVoxel = true;
                isMyColliderMesh = false;
            }
            else
            {
                myColliders = myMeshColliders;
                if (myColliders.Length > 0)
                {
                    isMyColliderMesh = true;
                    isMyColliderVoxel = false;
                }
                else
                {
                    isMyColliderMesh = false;
                    isMyColliderVoxel = false;
                }
            }

            mySphereColliders = GetComponents<SphereCollider>();
            myBoxColliders = GetComponents<BoxCollider>();
            myCapsuleColliders = GetComponents<CapsuleCollider>();

            myColliders = GetComponents<Collider>();
            List<PhysicsMaterial> origMaterials = new List<PhysicsMaterial>();
            for (int i = 0; i < myColliders.Length; ++i)
            {
                // Friction needs a relativistic correction, so we need variable PhysicMaterial parameters.
                Collider collider = myColliders[i];
                origMaterials.Add(collider.material);
                collider.material = Instantiate(collider.material);
            }
            origPhysicMaterials = origMaterials.ToArray();
        }

        public void UpdateColliderPosition()
        {
            if (isMyColliderVoxel || isNonrelativisticShader || (myColliders == null) || (myColliders.Length == 0))
            {
                return;
            }

            //If we have a MeshCollider and a compute shader, transform the collider verts relativistically:
            if (isMyColliderMesh && colliderShader && SystemInfo.supportsComputeShaders)
            {
                UpdateMeshCollider((MeshCollider)myColliders[0]);
            }
            //If we have a Collider, transform its center to its optical position
            else if (isMyColliderGeneral)
            {
                Vector4 aiw4 = GetComoving4Acceleration();

                int iTot = 0;
                for (int i = 0; i < mySphereColliders.Length; ++i)
                {
                    SphereCollider collider = mySphereColliders[i];
                    Vector3 pos = transform.TransformPoint((Vector4)colliderPiw[iTot]);
                    Vector3 pw = pos.WorldToOptical(peculiarVelocity, aiw4);
                    Vector3 testPos = transform.InverseTransformPoint(pw);
                    collider.center = testPos;
                    ++iTot;
                }

                for (int i = 0; i < myBoxColliders.Length; ++i)
                {
                    BoxCollider collider = myBoxColliders[i];
                    Vector3 pos = transform.TransformPoint((Vector4)colliderPiw[iTot]);
                    Vector3 pw = pos.WorldToOptical(peculiarVelocity, aiw4);
                    Vector3 testPos = transform.InverseTransformPoint(pw);
                    collider.center = testPos;
                    ++iTot;
                }

                for (int i = 0; i < myCapsuleColliders.Length; ++i)
                {
                    CapsuleCollider collider = myCapsuleColliders[i];
                    Vector3 pos = transform.TransformPoint((Vector4)colliderPiw[iTot]);
                    Vector3 pw = pos.WorldToOptical(peculiarVelocity, aiw4);
                    Vector3 testPos = transform.InverseTransformPoint(pw);
                    collider.center = testPos;
                    ++iTot;
                }
            }
        }
        #endregion

        #region Nonrelativistic shader
        private void SetUpContractor()
        {
            _localScale = transform.localScale;
            if (contractor)
            {
                contractor.parent = null;
                contractor.localScale = new Vector3(1, 1, 1);
                transform.parent = null;
                Destroy(contractor.gameObject);
            }
            GameObject contractorGO = new GameObject();
            contractorGO.name = gameObject.name + " Contractor";
            contractor = contractorGO.transform;
            contractor.parent = transform.parent;
            contractor.position = transform.position;
            transform.parent = contractor;
            transform.localPosition = Vector3.zero;
        }

        public void ContractLength()
        {
            Vector3 playerVel = state.PlayerVelocityVector;
            Vector3 relVel = (-playerVel).AddVelocity(peculiarVelocity);
            float relVelMag = relVel.sqrMagnitude;

            if (relVelMag > (state.MaxSpeed))
            {
                relVel.Normalize();
                relVelMag = state.MaxSpeed;
                relVel = relVelMag * relVel;
            }

            //Undo length contraction from previous state, and apply updated contraction:
            // - First, return to world frame:
            contractor.localScale = new Vector3(1, 1, 1);
            transform.localScale = _localScale;

            if (relVelMag > SRelativityUtil.FLT_EPSILON)
            {
                Quaternion rot = transform.rotation;

                relVelMag = Mathf.Sqrt(relVelMag);
                // - If we need to contract the object, unparent it from the contractor before rotation:
                //transform.parent = cparent;

                Quaternion origRot = transform.rotation;

                // - Rotate contractor to point parallel to velocity relative player:
                contractor.rotation = Quaternion.FromToRotation(Vector3.forward, relVel / relVelMag);

                // - Re-parent the object to the contractor before length contraction:
                transform.rotation = origRot;

                // - Set the scale based only on the velocity relative to the player:
                contractor.localScale = new Vector3(1, 1, 1).ContractLengthBy(relVelMag * Vector3.forward);
            }
        }

        public void UpdateContractorPosition()
        {
            if (!Application.isPlaying || !isNonrelativisticShader)
            {
                return;
            }

            if (!contractor)
            {
                SetUpContractor();
            }

            contractor.position = opticalPiw;
            transform.localPosition = Vector3.zero;
            ContractLength();
        }
        #endregion

        #region RelativisticObject internals

        // Get the start time of our object, so that we know where not to draw it
        public void SetStartTime()
        {
            Vector3 playerPos = state.playerTransform.position;
            float timeDelayToPlayer = Mathf.Sqrt((opticalPiw - playerPos).sqrMagnitude / state.SpeedOfLightSqrd);
            timeDelayToPlayer *= GetTimeFactor();
            StartTime = state.TotalTimeWorld - timeDelayToPlayer;
            hasStarted = false;
            if (myRenderer)
                myRenderer.enabled = false;
        }

        //Set the death time, so that we know at what point to destroy the object in the player's view point.
        public virtual void SetDeathTime()
        {
            Vector3 playerPos = state.playerTransform.position;
            float timeDelayToPlayer = Mathf.Sqrt((opticalPiw - playerPos).sqrMagnitude / state.SpeedOfLightSqrd);
            timeDelayToPlayer *= GetTimeFactor();
            DeathTime = state.TotalTimeWorld - timeDelayToPlayer;
        }
        public void ResetDeathTime()
        {
            DeathTime = float.PositiveInfinity;
        }

        void CombineParent()
        {
            if (GetComponent<ObjectMeshDensity>())
            {
                GetComponent<ObjectMeshDensity>().enabled = false;
            }
            bool wasStatic = gameObject.isStatic;
            gameObject.isStatic = false;
            int vertCount = 0, triangleCount = 0;
            Matrix4x4 worldLocalMatrix = transform.worldToLocalMatrix;

            //This code combines the meshes of children of parent objects
            //This increases our FPS by a ton
            //Get an array of the meshfilters
            MeshFilter[] meshFilters = GetComponentsInChildren<MeshFilter>(true);
            //Count submeshes
            int[] subMeshCount = new int[meshFilters.Length];
            //Get all the meshrenderers
            MeshRenderer[] meshRenderers = GetComponentsInChildren<MeshRenderer>(true);
            //Length of our original array
            int meshFilterLength = meshFilters.Length;
            //And a counter
            int subMeshCounts = 0;
            //We can optimize further for duplicate materials:
            Dictionary<string, Material> uniqueMaterials = new Dictionary<string, Material>();
            List<string> uniqueMaterialNames = new List<string>();

            //For every meshfilter,
            for (int y = 0; y < meshFilterLength; ++y)
            {
                //If it's null, ignore it.
                if (!meshFilters[y]) continue;
                if (!meshFilters[y].sharedMesh) continue;
                if (!meshFilters[y].sharedMesh.isReadable) continue;
                //else add its vertices to the vertcount
                vertCount += meshFilters[y].sharedMesh.vertices.Length;
                //Add its triangles to the count
                triangleCount += meshFilters[y].sharedMesh.triangles.Length;
                //Add the number of submeshes to its spot in the array
                subMeshCount[y] = meshFilters[y].mesh.subMeshCount;
                //And add up the total number of submeshes
                subMeshCounts += meshFilters[y].mesh.subMeshCount;
            }
            // Get a temporary array of EVERY vertex
            Vector3[] tempVerts = new Vector3[vertCount];
            //And make a triangle array for every submesh
            int[][] tempTriangles = new int[subMeshCounts][];

            for (int u = 0; u < subMeshCounts; ++u)
            {
                //Make every array the correct length of triangles
                tempTriangles[u] = new int[triangleCount];
            }
            //Also grab our UV texture coordinates
            Vector2[] tempUVs = new Vector2[vertCount];
            //And store a number of materials equal to the number of submeshes.
            Material[] tempMaterials = new Material[subMeshCounts];

            int vertIndex = 0;
            Mesh MFs;
            int subMeshIndex = 0;
            //For all meshfilters
            for (int i = 0; i < meshFilterLength; ++i)
            {
                //just doublecheck that the mesh isn't null
                MFs = meshFilters[i].sharedMesh;
                if (!MFs) continue;
                if (!MFs.isReadable) continue;

                //Otherwise, for all submeshes in the current mesh
                for (int q = 0; q < subMeshCount[i]; ++q)
                {
                    //turn off the original renderer
                    meshRenderers[i].enabled = false;
                    RelativisticObject ro = meshRenderers[i].GetComponent<RelativisticObject>();
                    if (ro)
                    {
                        ro.hasParent = true;
                    }
                    //grab its material
                    tempMaterials[subMeshIndex] = meshRenderers[i].materials[q];
                    //Check if material is unique
                    string name = meshRenderers[i].materials[q].name.Replace(" (Instance)", "");
                    if (!uniqueMaterials.ContainsKey(name))
                    {
                        uniqueMaterials.Add(name, meshRenderers[i].materials[q]);
                        uniqueMaterialNames.Add(name);
                    }
                    //Grab its triangles
                    int[] tempSubTriangles = MFs.GetTriangles(q);
                    //And put them into the submesh's triangle array
                    for (int k = 0; k < tempSubTriangles.Length; ++k)
                    {
                        tempTriangles[subMeshIndex][k] = tempSubTriangles[k] + vertIndex;
                    }
                    //Increment the submesh index
                    ++subMeshIndex;
                }
                Matrix4x4 cTrans = worldLocalMatrix * meshFilters[i].transform.localToWorldMatrix;
                //For all the vertices in the mesh
                for (int v = 0; v < MFs.vertices.Length; ++v)
                {
                    //Get the vertex and the UV coordinate
                    tempVerts[vertIndex] = cTrans.MultiplyPoint3x4(MFs.vertices[v]);
                    tempUVs[vertIndex] = MFs.uv[v];
                    ++vertIndex;
                }
            }

            //Put it all together now.
            Mesh myMesh = new Mesh();
            //If any materials are the same, we can combine triangles and give them the same material.
            myMesh.subMeshCount = uniqueMaterials.Count;
            myMesh.vertices = tempVerts;
            Material[] finalMaterials = new Material[uniqueMaterials.Count];
            for (int i = 0; i < uniqueMaterialNames.Count; ++i)
            {
                string uniqueName = uniqueMaterialNames[i];
                List<int> combineTriangles = new List<int>();
                for (int j = 0; j < tempMaterials.Length; ++j)
                {
                    string name = tempMaterials[j].name.Replace(" (Instance)", "");
                    if (uniqueName.Equals(name))
                    {
                        combineTriangles.AddRange(tempTriangles[j]);
                    }
                }
                myMesh.SetTriangles(combineTriangles.ToArray(), i);
                finalMaterials[i] = uniqueMaterials[uniqueMaterialNames[i]];
            }
            //Just shunt in the UV coordinates, we don't need to change them
            myMesh.uv = tempUVs;
            //THEN totally replace our object's mesh with this new, combined mesh

            MeshFilter meshy = gameObject.GetComponent<MeshFilter>();
            if (!GetComponent<MeshFilter>())
            {
                gameObject.AddComponent<MeshRenderer>();
                meshy = gameObject.AddComponent<MeshFilter>();
            }
            meshy.mesh = myMesh;

            GetComponent<MeshRenderer>().enabled = false;

            meshy.mesh.RecalculateNormals();
            if (uniqueMaterials.Count == 1)
            {
                meshy.GetComponent<Renderer>().material = finalMaterials[0];
            }
            else
            {
                meshy.GetComponent<Renderer>().materials = finalMaterials;
            }

            MeshCollider mCollider = GetComponent<MeshCollider>();
            if (mCollider)
            {
                mCollider.sharedMesh = myMesh;
            }

            transform.gameObject.SetActive(true);
            gameObject.isStatic = wasStatic;

            if (isCombinedColliderParent)
            {
                MeshCollider myMeshCollider = GetComponent<MeshCollider>();
                if (myMeshCollider)
                {
                    myMeshCollider.sharedMesh = myMesh;
                }  
            }
            else
            {
                MeshCollider[] childrenColliders = GetComponentsInChildren<MeshCollider>();
                List<Collider> dupes = new List<Collider>();
                for (int i = 0; i < childrenColliders.Length; ++i)
                {
                    MeshCollider orig = childrenColliders[i];
                    MeshCollider dupe = CopyComponent(childrenColliders[i], gameObject);
                    dupe.convex = orig.convex;
                    dupe.sharedMesh = Instantiate(orig.sharedMesh);
                    dupes.Add(dupe);
                }
                if (myColliders == null)
                {
                    myColliders = dupes.ToArray();
                }
                else
                {
                    dupes.AddRange(myColliders);
                    myColliders = dupes.ToArray();
                }
            }
            //"Delete" all children.
            for (int i = 0; i < transform.childCount; ++i)
            {
                GameObject child = transform.GetChild(i).gameObject;
                if (child.tag != "Contractor" && child.tag != "Voxel Collider")
                {
                    transform.GetChild(i).gameObject.SetActive(false);
                    Destroy(transform.GetChild(i).gameObject);
                }
            }
            GetComponent<MeshRenderer>().enabled = true;
            GetComponent<RelativisticObject>().enabled = true;
        }

        T CopyComponent<T>(T original, GameObject destination) where T : Component
        {
            System.Type type = original.GetType();
            Component copy = destination.AddComponent(type);
            System.Reflection.FieldInfo[] fields = type.GetFields();
            foreach (System.Reflection.FieldInfo field in fields)
            {
                field.SetValue(copy, field.GetValue(original));
            }
            return copy as T;
        }

        private void UpdateShaderParams()
        {
            if (!Application.isPlaying || !myRenderer)
            {
                return;
            }

            //Send our object's v/c (Velocity over the Speed of Light) to the shader

            Vector3 tempViw = peculiarVelocity / state.SpeedOfLight;
            Vector4 tempPao = GetComoving4Acceleration();
            Vector4 tempVr = (-state.PlayerVelocityVector).AddVelocity(peculiarVelocity) / state.SpeedOfLight;

            //Velocity of object Lorentz transforms are the same for all points in an object,
            // so it saves redundant GPU time to calculate them beforehand.
            Matrix4x4 viwLorentzMatrix = SRelativityUtil.GetLorentzTransformMatrix(tempViw);

            // Metric default (doesn't have correct signature):
            Matrix4x4 intrinsicMetric = state.conformalMap.GetMetric(piw);

            colliderShaderParams.viw = tempViw;
            colliderShaderParams.pao = tempPao;
            colliderShaderParams.viwLorentzMatrix = viwLorentzMatrix;
            colliderShaderParams.invViwLorentzMatrix = viwLorentzMatrix.inverse;
            for (int i = 0; i < myRenderer.materials.Length; ++i)
            {
                myRenderer.materials[i].SetVector("_viw", tempViw);
                myRenderer.materials[i].SetVector("_pao", tempPao);
                myRenderer.materials[i].SetMatrix("_viwLorentzMatrix", viwLorentzMatrix);
                myRenderer.materials[i].SetMatrix("_invViwLorentzMatrix", viwLorentzMatrix.inverse);
                myRenderer.materials[i].SetMatrix("_intrinsicMetric", intrinsicMetric);
                myRenderer.materials[i].SetMatrix("_invIntrinsicMetric", intrinsicMetric.inverse);
                myRenderer.materials[i].SetVector("_vr", tempVr);
                myRenderer.materials[i].SetFloat("_lastUpdateSeconds", Time.time);
            }
        }

        private void UpdateBakingShaderParams()
        {
            if (!myRenderer)
            {
                return;
            }

            //Send our object's v/c (Velocity over the Speed of Light) to the shader

            Vector3 tempViw = peculiarVelocity / state.SpeedOfLight;
            Vector4 tempPao = GetComoving4Acceleration();
            Vector4 tempVr = (-state.PlayerVelocityVector).AddVelocity(peculiarVelocity) / state.SpeedOfLight;

            //Velocity of object Lorentz transforms are the same for all points in an object,
            // so it saves redundant GPU time to calculate them beforehand.
            Matrix4x4 viwLorentzMatrix = SRelativityUtil.GetLorentzTransformMatrix(tempViw);

            // Metric default (doesn't have correct signature):
            Matrix4x4 intrinsicMetric = state.conformalMap.GetMetric(piw);

            colliderShaderParams.viw = tempViw;
            colliderShaderParams.pao = tempPao;
            colliderShaderParams.viwLorentzMatrix = viwLorentzMatrix;
            colliderShaderParams.invViwLorentzMatrix = viwLorentzMatrix.inverse;
            for (int i = 0; i < myRenderer.sharedMaterials.Length; ++i)
            {
                if (myRenderer.sharedMaterials[i] == null) {
                    continue;
                }
                myRenderer.sharedMaterials[i] = Instantiate(myRenderer.sharedMaterials[i]);
                myRenderer.sharedMaterials[i].SetVector("_viw", tempViw);
                myRenderer.sharedMaterials[i].SetVector("_pao", tempPao);
                myRenderer.sharedMaterials[i].SetMatrix("_viwLorentzMatrix", viwLorentzMatrix);
                myRenderer.sharedMaterials[i].SetMatrix("_invViwLorentzMatrix", viwLorentzMatrix.inverse);
                myRenderer.sharedMaterials[i].SetMatrix("_intrinsicMetric", intrinsicMetric);
                myRenderer.sharedMaterials[i].SetMatrix("_invIntrinsicMetric", intrinsicMetric.inverse);
                myRenderer.sharedMaterials[i].SetVector("_vr", tempVr);
                myRenderer.sharedMaterials[i].SetFloat("_lastUpdateSeconds", Time.time);
            }
        }

        public void KillObject()
        {
            gameObject.SetActive(false);
            //Destroy(this.gameObject);
        }

        //This is a function that just ensures we're slower than our maximum speed. The VIW that Unity sets SHOULD (it's creator-chosen) be smaller than the maximum speed.
        private void checkSpeed()
        {
            if (isLightMapStatic)
            {
                return;
            }

            float maxSpeed = state.MaxSpeed - 0.01f;
            float maxSpeedSqr = maxSpeed * maxSpeed;

            if (peculiarVelocity.sqrMagnitude > maxSpeedSqr)
            {
                peculiarVelocity = peculiarVelocity.normalized * maxSpeed;
            }
            
            if (trnsfrmdMeshVerts == null)
            {
                return;
            }

            // The tangential velocities of each vertex should also not be greater than the maximum speed.
            // (This is a relatively computationally costly check, but it's good practice.

            for (int i = 0; i < trnsfrmdMeshVerts.Length; ++i)
            {
                Vector3 disp = Vector3.Scale(trnsfrmdMeshVerts[i], transform.lossyScale);
                Vector3 tangentialVel = Vector3.Cross(aviw, disp);
                float tanVelMagSqr = tangentialVel.sqrMagnitude;
                if (tanVelMagSqr > maxSpeedSqr)
                {
                    aviw = aviw.normalized * maxSpeed / disp.magnitude;
                }
            }
        }

        private void UpdateRigidbodyVelocity()
        {
            float gamma = GetTimeFactor();

            if (myRigidbody && !myRigidbody.isKinematic)
            {
                // If movement is frozen, set to zero.
                // If we're in an invalid state, (such as before full initialization,) set to zero.
                if (state.isMovementFrozen)
                {
                    myRigidbody.linearVelocity = Vector3.zero;
                    myRigidbody.angularVelocity = Vector3.zero;
                }
                else
                {
                    // Factor of gamma corrects for length-contraction, (which goes like 1/gamma).
                    // Effectively, this replaces Time.DeltaTime with Time.DeltaTime / gamma.
                    myRigidbody.linearVelocity = gamma * peculiarVelocity;
                    myRigidbody.angularVelocity = gamma * aviw;

                    Vector3 properAccel = nonGravAccel + leviCivitaDevAccel;
                    if (comoveViaAcceleration)
                    {
                        // This is not actually "proper acceleration," with this option active.
                        properAccel += state.conformalMap.GetRindlerAcceleration(piw);
                    }
                    if (properAccel.sqrMagnitude > SRelativityUtil.FLT_EPSILON)
                    {
                        myRigidbody.AddForce(gamma * properAccel, ForceMode.Acceleration);
                    }
                }

                nonGravAccel = Vector3.zero;

                // Factor of 1/gamma corrects for time-dilation, (which goes like gamma).
                // Unity's (physically inaccurate) drag formula is something like,
                // velocity = velocity * (1 - drag * Time.deltaTime),
                // where we counterbalance the time-dilation factor above, for observer path invariance.
                myRigidbody.linearDamping = unityDrag / gamma;
                myRigidbody.angularDamping = unityAngularDrag / gamma;
            } else {
                nonGravAccel = Vector3.zero;
            }

            if (myColliders == null) {
                return;
            }

            for (int i = 0; i < myColliders.Length; ++i)
            {
                Collider collider = myColliders[i];

                // Energy dissipation goes like mu * F_N * d.
                // If the path parallel to d is also parallel to relative velocity,
                // d "already looks like" d' / gamma, to player, so multiply gamma * mu.
                // If the path parallel to d is perpendicular to relative velocity,
                // F_N "already looks like" it's being applied in time t' * gamma, to player, so multiply gamma * mu.
                collider.material.staticFriction = gamma * origPhysicMaterials[i].staticFriction;
                collider.material.dynamicFriction = gamma * origPhysicMaterials[i].dynamicFriction;
                // rapidity_after / rapidity_before - Doesn't seem to need an adjustment.
                collider.material.bounciness = origPhysicMaterials[i].bounciness;
            }
        }
        #endregion

        #region Unity lifecycle
        protected void OnDestroy()
        {
            if (paramsBuffer != null) paramsBuffer.Release();
            if (vertBuffer != null) vertBuffer.Release();
            if (Application.isPlaying && (contractor != null)) Destroy(contractor.gameObject);
        }

        protected void OnEnable() {
            //Also get the meshrenderer so that we can give it a unique material
            if (myRenderer == null)
            {
                myRenderer = GetComponent<Renderer>();
            }
            //If we have a MeshRenderer on our object and it's not world-static
            if (myRenderer)
            {
                UpdateBakingShaderParams();
            }
        }

        protected void Awake()
        {
            _localScale = transform.localScale;
            myRigidbody = GetComponent<Rigidbody>();
        }

        protected void Start()
        {
            hasStarted = false;
            isPhysicsUpdateFrame = false;

            if (myRigidbody)
            {
                myRigidbody.linearDamping = unityDrag;
                myRigidbody.angularDamping = unityAngularDrag;
                baryonCount = mass * SRelativityUtil.avogadroNumber / initialAverageMolarMass;
            }
            
            _piw = isNonrelativisticShader ? transform.position.OpticalToWorld(peculiarVelocity, GetComoving4Acceleration()) : transform.position;
            riw = transform.rotation;
            checkSpeed();
            UpdatePhysicsCaches();

            if (isNonrelativisticShader)
            {
                UpdateContractorPosition();
            }
            else
            {
                transform.position = piw;
            }

            // Update the shader parameters if necessary
            UpdateShaderParams();

            wasKinematic = false;
            wasFrozen = false;

            UpdateColliders();

            MarkStaticColliderPos();

            //Get the meshfilter
            if (isParent)
            {
                CombineParent();
            }

            meshFilter = GetComponent<MeshFilter>();

            if (myRigidbody)
            {
                //Native rigidbody gravity should not be used except during isFullPhysX.
                myRigidbody.useGravity = useGravity && !isLightMapStatic;
            }
        }

        protected bool isPhysicsCacheValid;

        protected void UpdatePhysicsCaches()
        {
            isPhysicsCacheValid = false;

            updateMetric = GetMetric();
            updatePlayerViwTimeFactor = state.PlayerVelocityVector.InverseGamma(updateMetric);
            updateViwTimeFactor = viw.InverseGamma(updateMetric);
            updateWorld4Acceleration = comovingAccel.ProperToWorldAccel(viw, updateViwTimeFactor);
            updateTisw = piw.GetTisw(viw, updateWorld4Acceleration);

            isPhysicsCacheValid = true;
        }

        protected void AfterPhysicsUpdate()
        {
            oldLocalTimeFixedUpdate = GetLocalTime();

            if (myRigidbody)
            {
                if (!isNonrelativisticShader)
                {
                    // Get the relativistic position and rotation after the physics update:
                    riw = myRigidbody.rotation;
                    _piw = myRigidbody.position;
                }

                // Now, update the velocity and angular velocity based on the collision result:
                _aviw = myRigidbody.angularVelocity / updatePlayerViwTimeFactor;
                peculiarVelocity = myRigidbody.linearVelocity.RapidityToVelocity(updateMetric);
            }

            if (isNonrelativisticShader)
            {
                // Immediately correct the nonrelativistic shader position.
                UpdateContractorPosition();
            }

            if (isMonopoleAccel)
            {
                float softenFactor = 1 + monopoleAccelerationSoften;
                float tempSoftenFactor = Mathf.Pow(softenFactor, 1.0f / 4);

                monopoleTemperature /= tempSoftenFactor;
                float origBackgroundTemp = state.gravityBackgroundPlanckTemperature;
                state.gravityBackgroundPlanckTemperature /= tempSoftenFactor;

                Vector3 accel = ((peculiarVelocity - oldVelocity) / lastFixedUpdateDeltaTime + aiw) / softenFactor;
                EvaporateMonopole(softenFactor * lastFixedUpdateDeltaTime, accel);

                state.gravityBackgroundPlanckTemperature = origBackgroundTemp;
                monopoleTemperature *= tempSoftenFactor;
            }

            checkSpeed();
            UpdatePhysicsCaches();
            UpdateColliderPosition();
        }

        protected void Update()
        {
            if (isPhysicsUpdateFrame)
            {
                AfterPhysicsUpdate();
            }
            isPhysicsUpdateFrame = false;
        }

        protected void LateUpdate()
        {
            oldLocalTimeUpdate = GetLocalTime();
        }

        protected void FixedUpdate()
        {
            if (isPhysicsUpdateFrame)
            {
                AfterPhysicsUpdate();
            }
            isPhysicsUpdateFrame = false;

            if (!isPhysicsCacheValid)
            {
                UpdatePhysicsCaches();
            }

            if (state.isMovementFrozen || !state.isInitDone)
            {
                // If our rigidbody is not null, and movement is frozen, then set the object to standstill.
                UpdateRigidbodyVelocity();
                UpdateShaderParams();

                // We're done.
                return;
            }

            float deltaTime = state.FixedDeltaTimePlayer * GetTimeFactor();
            localTimeOffset += deltaTime - state.FixedDeltaTimeWorld;

            if (isLightMapStatic)
            {
                if (isMonopoleAccel)
                {
                    EvaporateMonopole(deltaTime, aiw);
                }

                UpdateColliderPosition();

                return;
            }

            if (!comoveViaAcceleration)
            {
                Comovement cm = state.conformalMap.ComoveOptical(deltaTime, piw, riw);
                riw = cm.riw;
                _piw = cm.piw;

                if (!isNonrelativisticShader && myRigidbody)
                {
                    // We'll MovePosition() for isNonrelativisticShader, further below.
                    myRigidbody.MovePosition(piw);
                }
            }

            //As long as our object is actually alive, perform these calculations
            if (meshFilter && transform) 
            {
                //If we're past our death time (in the player's view, as seen by tisw)
                if (state.TotalTimeWorld + localTimeOffset + deltaTime > DeathTime)
                {
                    KillObject();
                }
                else if (!hasStarted && (state.TotalTimeWorld + localTimeOffset + deltaTime > StartTime))
                {
                    hasStarted = true;
                    //Grab our renderer.
                    Renderer tempRenderer = GetComponent<Renderer>();
                    if (!tempRenderer.enabled)
                    {
                        tempRenderer.enabled = !hasParent;
                        AudioSource[] audioSources = GetComponents<AudioSource>();
                        if (audioSources.Length > 0)
                        {
                            for (int i = 0; i < audioSources.Length; ++i)
                            {
                                audioSources[i].enabled = true;
                            }
                        }
                    }
                }
            }

            #region rigidbody
            // The rest of the updates are for objects with Rigidbodies that move and aren't asleep.
            if (isKinematic || !myRigidbody || myRigidbody.IsSleeping())
            {
                if (myRigidbody && !myRigidbody.isKinematic)
                {
                    myRigidbody.angularVelocity = Vector3.zero;
                    myRigidbody.linearVelocity = Vector3.zero;

                    if (!isKinematic)
                    {
                        _aviw = Vector3.zero;
                        UpdateMotion(Vector3.zero, Vector3.zero);
                    }
                }

                if (!isLightMapStatic)
                {
                    transform.position = isNonrelativisticShader ? opticalPiw : piw;
                }

                UpdateShaderParams();

                // We're done.
                return;
            }

            if (isNonrelativisticShader)
            {
                // Update riw
                float aviwMag = aviw.magnitude;
                Quaternion diffRot;
                if (aviwMag <= SRelativityUtil.FLT_EPSILON)
                {
                    diffRot = Quaternion.identity;
                }
                else
                {
                    diffRot = Quaternion.AngleAxis(Mathf.Rad2Deg * deltaTime * aviwMag, aviw / aviwMag);
                }
                riw = riw * diffRot;
                myRigidbody.MoveRotation(riw);

                // Update piw from "peculiar velocity" in free fall coordinates.
                _piw += deltaTime * peculiarVelocity;

                transform.parent = null;
                myRigidbody.MovePosition(opticalPiw);
                contractor.position = myRigidbody.position;
                transform.parent = contractor;
                transform.localPosition = Vector3.zero;
                ContractLength();
            }
            UpdateColliderPosition();
            #endregion

            // FOR THE PHYSICS UPDATE ONLY, we give our rapidity to the Rigidbody
            UpdateRigidbodyVelocity();

            oldVelocity = peculiarVelocity;
            lastFixedUpdateDeltaTime = deltaTime;

            isPhysicsUpdateFrame = true;
        }

        protected void EvaporateMonopole(float deltaTime, Vector3 myAccel)
        {
            // If the RelativisticObject is at rest on the ground, according to Strano 2019, (not yet peer reviewed,)
            // it loses surface acceleration, (not weight force, directly,) the longer it stays in this configuration.
            // The Rindler horizon evaporates as would Schwarzschild, for event horizon surface acceleration equivalent
            // between the Rindler and Schwarzschild metrics. Further, Hawking(-Unruh, et al.) acceleration might have
            // the same effect.

            // The Rindler horizon evaporates as a Schwarzschild event horizon with the same surface gravity, according to Strano.
            // We add any background radiation power, proportional to the fourth power of the background temperature.
            double alpha = myAccel.magnitude;
            bool isNonZeroTemp = alpha > SRelativityUtil.FLT_EPSILON;

            double r = double.PositiveInfinity;
            // If alpha is in equilibrium with the background temperature, there is no evaporation.
            if (isNonZeroTemp)
            {
                // Surface acceleration at event horizon:
                r = state.SpeedOfLightSqrd / (2 * alpha);
                r = SRelativityUtil.EffectiveRaditiativeRadius((float)r, state.gravityBackgroundPlanckTemperature);
            }

            if (r < state.planckLength)
            {
                // For minimum area calculation, below.
                r = state.planckLength;
            }

            if (!double.IsInfinity(r) && !double.IsNaN(r))
            {
                isNonZeroTemp = true;
                r += SRelativityUtil.SchwarzschildRadiusDecay(deltaTime, r);
                if (r <= SRelativityUtil.FLT_EPSILON)
                {
                    leviCivitaDevAccel += myAccel;
                }
                else
                {
                    double alphaF = state.SpeedOfLightSqrd / (2 * r);
                    leviCivitaDevAccel += (float)(alpha - alphaF) * myAccel.normalized;
                }
            }

            if (myRigidbody)
            {
                double myTemperature = monopoleTemperature;
                double surfaceArea;
                if (meshFilter == null)
                {
                    Vector3 lwh = transform.localScale;
                    surfaceArea = 2 * (lwh.x * lwh.y + lwh.x * lwh.z + lwh.y * lwh.z);
                }
                else
                {
                    surfaceArea = meshFilter.sharedMesh.SurfaceArea();
                }
                // This is the ambient temperature, including contribution from comoving accelerated rest temperature.
                double ambientTemperature = isNonZeroTemp ? SRelativityUtil.SchwarzRadiusToPlanckScaleTemp(r) : state.gravityBackgroundPlanckTemperature;
                double dm = (gravitonEmissivity * surfaceArea * SRelativityUtil.sigmaPlanck * (Math.Pow(myTemperature, 4) - Math.Pow(ambientTemperature, 4))) / state.planckArea;

                // Momentum is conserved. (Energy changes.)
                Vector3 momentum = mass * peculiarVelocity;

                double camm = (mass - dm) * SRelativityUtil.avogadroNumber / baryonCount;

                if ((myTemperature >= 0) && (camm < fundamentalAverageMolarMass))
                {
                    currentAverageMolarMass = fundamentalAverageMolarMass;
                }
                else if (camm <= 0)
                {
                    mass = 0;
                }
                else
                {
                    mass -= (float)dm;
                }

                if (mass > SRelativityUtil.FLT_EPSILON)
                {
                    peculiarVelocity = momentum / mass;
                }
            }
        }

        public void OnCollisionEnter(Collision collision)
        {
            OnCollision();
        }

        public void OnCollisionStay(Collision collision)
        {
            OnCollision();
        }

        public void OnCollisionExit(Collision collision)
        {
            
        }
        #endregion

        #region Rigidbody mechanics
        public void OnCollision()
        {
            if (isKinematic || state.isMovementFrozen || !myRigidbody)
            {
                return;
            }

            // Like how Rigidbody components are co-opted for efficient relativistic motion,
            // it's feasible to get (at least reasonable, if not exact) relativistic collision
            // handling by transforming the end state after PhysX collisions.

            // We pass the RelativisticObject's rapidity to the rigidbody, right before the physics update
            // We restore the time-dilated visual apparent velocity, afterward

            if (isNonrelativisticShader)
            {
                riw = myRigidbody.rotation;
                _piw = myRigidbody.position.OpticalToWorld(peculiarVelocity, updateWorld4Acceleration);
            }

            isPhysicsUpdateFrame = true;
            AfterPhysicsUpdate();
            isPhysicsUpdateFrame = false;
        }
        #endregion
    }
}
