#ifndef SIMPLE_2D_ACTUATOR_LINE_H
#define SIMPLE_2D_ACTUATOR_LINE_H

#include "amr-wind/equation_systems/icns/MomentumSource.H"
#include "amr-wind/core/SimTime.H"

namespace amr_wind::pde::icns {

/** A simple 2D actuator lifting line that can translate in time.
 * \ingroup icns_src
 *
 * Implemented by M. Churchfield to understand velocity sampling
 * and force application position with respect to AMR-Wind's
 * time-stepping algorithm to inform the full actuator line 
 * implementation.
 */

class Simple2DActuatorLine : public MomentumSource::Register<Simple2DActuatorLine>
{
public:
    static std::string identifier() { return "Simple2DActuatorLine"; }

    explicit Simple2DActuatorLine(const CFDSim& sim);

    ~Simple2DActuatorLine() override;

    void operator()(
        const int lev,
        const amrex::MFIter& mfi,
        const amrex::Box& bx,
        const FieldState fstate,
        const amrex::Array4<amrex::Real>& src_term) const override;

private:
    const SimTime& m_time;

    const amrex::AmrCore& m_mesh;

    const Field& m_velocity;

    //! Actuator initial position.
    amrex::Vector<amrex::Real> m_actInitPos{{0.0, 0.0, 0.0}};

    //! Actuator translation velocity.
    amrex::Vector<amrex::Real> m_actVelTrans{{0.0, 0.0, 0.0}};

    //! Angle of incidence of modeled wing (degrees).
    amrex::Real m_actAngleInc{8.0};

    //! Gaussian body force characteristic width.
    amrex::Real m_actEps{1.0};

    //! Direction of line (x=0, y=1, or z=2).
    int m_actAxis{1};

    //! Actuator position at current time.
    amrex::Vector<amrex::Real> m_actPos{{0.0, 0.0, 0.0}};

    //! Current value of coefficient of lift.
    amrex::Real m_Cl{0.0};

    //! Current actuator force vector.
    amrex::Vector<amrex::Real> m_forceVector{{0.0, 0.0, 0.0}};
};

} // namespace amr_wind::pde::icns

#endif
