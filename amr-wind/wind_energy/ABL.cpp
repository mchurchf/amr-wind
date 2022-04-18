#include <memory>

#include "amr-wind/wind_energy/ABL.H"
#include "amr-wind/wind_energy/ABLFieldInit.H"
#include "amr-wind/wind_energy/ABLBoundaryPlane.H"
#include "amr-wind/equation_systems/icns/source_terms/ABLForcing.H"
#include "amr-wind/equation_systems/icns/source_terms/ABLMeanBoussinesq.H"
#include "amr-wind/incflo.H"

#include "AMReX_ParmParse.H"
#include "AMReX_MultiFab.H"
#include <zmq.h>

namespace amr_wind {

ABL::ABL(CFDSim& sim)
    : m_sim(sim)
    , m_velocity(sim.pde_manager().icns().fields().field)
    , m_mueff(sim.pde_manager().icns().fields().mueff)
    , m_density(sim.repo().get_field("density"))
    , m_abl_wall_func(sim)
{
    // Register temperature equation
    // FIXME: this should be optional?
    auto& teqn = sim.pde_manager().register_transport_pde("Temperature");
    m_temperature = &(teqn.fields().field);

    {
        std::string statistics_mode = "precursor";
        int dir = 2;
        amrex::ParmParse pp("ABL");
        pp.query("enable_hybrid_rl_mode", m_hybrid_rl);
        pp.query("initial_sdr_value", m_init_sdr);
        pp.query("normal_direction", dir);
        pp.query("statistics_mode", statistics_mode);
        m_stats =
            ABLStatsBase::create(statistics_mode, sim, m_abl_wall_func, dir);
    }

    // Instantiate the ABL field initializer
    m_field_init = std::make_unique<ABLFieldInit>();

    // Instantiate the ABL boundary plane IO
    m_bndry_plane = std::make_unique<ABLBoundaryPlane>(sim);
}

ABL::~ABL() = default;

/** Initialize the velocity and temperature fields at the beginning of the
 *  simulation.
 *
 *  \sa amr_wind::ABLFieldInit
 */
void ABL::initialize_fields(int level, const amrex::Geometry& geom)
{
    auto& velocity = m_velocity(level);
    auto& density = m_density(level);
    auto& temp = (*m_temperature)(level);

    if (m_field_init->add_temperature_perturbations()) {
        m_field_init->perturb_temperature(level, geom, *m_temperature);
    } else {
        temp.setVal(0.0);
    }

    for (amrex::MFIter mfi(density); mfi.isValid(); ++mfi) {
        const auto& vbx = mfi.validbox();

        (*m_field_init)(
            vbx, geom, velocity.array(mfi), density.array(mfi),
            temp.array(mfi));
    }

    if (m_sim.repo().field_exists("tke")) {
        m_tke = &(m_sim.repo().get_field("tke"));
        auto& tke = (*m_tke)(level);
        m_field_init->init_tke(geom, tke);
    }
}

void ABL::post_init_actions()
{
    if (m_hybrid_rl) {
        m_sdr = &(m_sim.repo().get_field("sdr"));
        m_sdr->setVal(m_init_sdr);
    }

    m_stats->post_init_actions();

    m_abl_wall_func.init_log_law_height();

    m_abl_wall_func.update_umean(
        m_stats->vel_profile(), m_stats->theta_profile());

    // Register ABL wall function for velocity
    m_velocity.register_custom_bc<ABLVelWallFunc>(m_abl_wall_func);
    (*m_temperature).register_custom_bc<ABLTempWallFunc>(m_abl_wall_func);

    m_bndry_plane->post_init_actions();
}

/** Perform tasks at the beginning of a new timestep
 *
 *  For ABL simulations this method invokes the FieldPlaneAveraging class to
 *  compute spatial averages at all z-levels on the coarsest mesh (level 0).
 *
 *  The spatially averaged velocity is used to determine the current mean
 *  velocity at the forcing height (if driving pressure gradient term is active)
 *  and also determines the average friction velocity for use in the ABL wall
 *  function computation.
 */
void ABL::pre_advance_work()
{
    const auto& vel_pa = m_stats->vel_profile();
    m_abl_wall_func.update_umean(
        m_stats->vel_profile(), m_stats->theta_profile());

    if (m_abl_forcing != nullptr) {
        const amrex::Real zh = m_abl_forcing->forcing_height();
        const amrex::Real vx = vel_pa.line_average_interpolated(zh, 0);
        const amrex::Real vy = vel_pa.line_average_interpolated(zh, 1);
        // Set the mean velocities at the forcing height so that the source
        // terms can be computed during the time integration calls



        // Wait for a response from control center
        // This response will include wind speed and direction
        char charFromControlCenter [9900]; // char array of what we got back from ControlCenter
        if(amrex::ParallelDescriptor::IOProcessor()){
            zmq_recv (m_sim.zmq_requester, charFromControlCenter, 9900, 0);
        }

        // Show what is in the response
        amrex::Print() << "initial response from control center: [" << charFromControlCenter << "] \n";

        amrex::Real wind_speed;
        amrex::Real wind_direction;

        // Unpack the values from the control center using a string stream
        std::stringstream ss(charFromControlCenter);
        ss >> wind_speed;  // This seems to work, I'm not positive it should?  wind_speed is a float but ss is a string?  maybe c++ magic?
        ss >> wind_direction; // This seems to work, I'm not positive it should?  wind_speed is a float but ss is a string?  maybe c++ magic?
        std::cout << "wind speed: " << wind_speed << "\n";
        std::cout << "wind direction: " << wind_direction << "\n";

        if(wind_direction > 180.0){
            wind_direction -= 180.0;
        }
        else{
            wind_direction += 180.0;
        }
        wind_direction = 90.0 - wind_direction;
        if(wind_direction < 0.0){
            wind_direction += 360.0;
        }

        const amrex::Real wind_direction_radian = utils::radians(wind_direction);
        amrex::Real tvx = wind_speed*std::cos(wind_direction_radian);
        amrex::Real tvy = wind_speed*std::sin(wind_direction_radian);
        amrex::Print() << "target velocities: " << tvx << ' ' << tvy << std::endl;

        amrex::ParallelDescriptor::Bcast(
            &tvx, 1,
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());

        amrex::ParallelDescriptor::Bcast(
            &tvy, 1,
            amrex::ParallelDescriptor::IOProcessorNumber(),
            amrex::ParallelDescriptor::Communicator());

        m_abl_forcing->set_target_velocities(tvx,tvy);

        m_abl_forcing->set_mean_velocities(vx, vy);

        m_abl_forcing->set_target_velocities(tvx,tvy);
        m_abl_forcing->set_mean_velocities(vx, vy);
    }

    if (m_abl_mean_bous != nullptr) {
        m_abl_mean_bous->mean_temperature_update(m_stats->theta_profile());
    }

    m_bndry_plane->pre_advance_work();
}

/** Perform tasks at the end of a new timestep
 *
 *  For ABL simulations, this method writes all plane-averaged profiles and
 *  integrated statistics to output
 */
void ABL::post_advance_work()
{
    m_stats->post_advance_work();
    m_bndry_plane->post_advance_work();

    // Declare an array of turbine powers
    const int num_turbines = 4;
    float turbine_power_array[num_turbines];
    amrex::Real wind_speed{0.0};

    amrex::Real vx{0.0};
    amrex::Real vy{0.0};

    if (m_abl_forcing != nullptr) {
        const amrex::Real zh = m_abl_forcing->forcing_height();
        const auto& vel_pa = m_stats->vel_profile();
        vx = vel_pa.line_average_interpolated(zh, 0);
        vy = vel_pa.line_average_interpolated(zh, 1);
        wind_speed = std::sqrt(vx*vx+vy*vy);
    }

    turbine_power_array[0] = wind_speed * wind_speed * wind_speed + 100.0;
    turbine_power_array[1] = wind_speed * wind_speed * wind_speed + 50.0;
    turbine_power_array[2] = wind_speed * wind_speed * wind_speed * 3/5 + 50;
    turbine_power_array[3] = wind_speed * wind_speed * wind_speed * 3/5;

    const amrex::Real wind_direction = std::atan2(vy,vx);
    std::stringstream ssToControlCenter; // stringStream used to construct strToSSC
    ssToControlCenter << m_sim.time().time_index() << " "; // First entry is timestep
    ssToControlCenter << wind_speed << " "; // First entry is wind_speed
    ssToControlCenter << utils::degrees(wind_direction); // First entry is wind_direction
    // Next the turbine information
    for (int i = 0; i < num_turbines; i++){
        ssToControlCenter << " " << turbine_power_array[i];
    }
    std::string strToControlCenter = ssToControlCenter.str(); //Convert to a string
    std::cout << "message to control center: [" << strToControlCenter << "] \n";

    // Send the data to the control center
    if(amrex::ParallelDescriptor::IOProcessor()){
        zmq_send (m_sim.zmq_requester, strToControlCenter.c_str(), 9900, 0);
    }



}

} // namespace amr_wind
