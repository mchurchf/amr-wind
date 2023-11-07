#include "amr-wind/equation_systems/icns/source_terms/Simple2DActuatorLine.H"
#include "amr-wind/CFDSim.H"
#include "amr-wind/utilities/trig_ops.H"

#include "AMReX_ParmParse.H"
#include "AMReX_Gpu.H"

namespace amr_wind::pde::icns {

Simple2DActuatorLine::Simple2DActuatorLine(const CFDSim& sim)
    : m_time(sim.time()), m_mesh(sim.mesh()), m_velocity(sim.repo().get_field("velocity"))
{
    // Read the Simple 2D Actuator Line parameters.
    amrex::ParmParse pp("Simple2DActuatorLine");

    pp.queryarr("initial_position", m_actInitPos, 0, AMREX_SPACEDIM);

    pp.queryarr("translational_velocity", m_actVelTrans, 0, AMREX_SPACEDIM);

    pp.get("wing_incidence_angle", m_actAngleInc);

    pp.get("line_axis", m_actAxis);

    pp.get("epsilon", m_actEps);
}

Simple2DActuatorLine::~Simple2DActuatorLine() = default;

void Simple2DActuatorLine::operator()(
    const int lev,
    const amrex::MFIter& mfi,
    const amrex::Box& bx,
    const FieldState fstate,
    const amrex::Array4<amrex::Real>& src_term) const
{
    // Get mesh information. 
    const auto& problo = m_mesh.Geom(lev).ProbLoArray();
    const auto& probhi = m_mesh.Geom(lev).ProbHiArray();
    const auto& dx = m_mesh.Geom(lev).CellSizeArray();

    // Get time information.
    const amrex::Real time = m_time.current_time();
    std::cout << "Time = " << time << std::endl;

    // Compute actuator position.
    

    // Make data available to GPU.   
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> actInitPos{
        {m_actInitPos[0], m_actInitPos[1], m_actInitPos[2]}};

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> actVelTrans{
        {m_actVelTrans[0], m_actVelTrans[1], m_actVelTrans[2]}};

    const amrex::Real actAngleInc = m_actAngleInc;

    const amrex::Real actEps = m_actEps;

    const int actAxis = m_actAxis;

    const auto& vel =
        m_velocity.state(field_impl::dof_state(fstate))(lev).const_array(mfi);

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const amrex::Real x = problo[0] + (i + 0.5) * dx[0];
        const amrex::Real y = problo[1] + (j + 0.5) * dx[1];
        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

        src_term(i, j, k, 0) += 
            0.0; 
        src_term(i, j, k, 1) += 
            0.0; 
        src_term(i, j, k, 2) += 
            0.0; 
    });


/*
    const amrex::Real tau = m_tau;
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> ref_vel{
        {m_ref_vel[0], m_ref_vel[1], m_ref_vel[2]}};
    const auto& vel =
        m_velocity.state(field_impl::dof_state(fstate))(lev).const_array(mfi);

    // Constants used to determine the fringe region coefficient
    const amrex::Real dRD = m_dRD;
    const amrex::Real dFull = m_dFull;

    // Which coordinate directions to force
    const amrex::Real fx = m_fcoord[0];
    const amrex::Real fy = m_fcoord[1];
    const amrex::Real fz = m_fcoord[2];

    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        amrex::Real coeff = 0.0;
        const amrex::Real z = problo[2] + (k + 0.5) * dx[2];

        if (probhi[2] - z > dRD + dFull) {
            coeff = 0.0;
        } else if (probhi[2] - z > dFull) {
            coeff = 0.5 * std::cos(M_PI * (probhi[2] - dFull - z) / dRD) + 0.5;
        } else {
            coeff = 1.0;
        }
        src_term(i, j, k, 0) +=
            fx * coeff * (ref_vel[0] - vel(i, j, k, 0)) / tau;
        src_term(i, j, k, 1) +=
            fy * coeff * (ref_vel[1] - vel(i, j, k, 1)) / tau;
        src_term(i, j, k, 2) +=
            fz * coeff * (ref_vel[2] - vel(i, j, k, 2)) / tau;
    });
*/
}

} // namespace amr_wind::pde::icns
