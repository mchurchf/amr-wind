#ifndef CONSOLE_IO_H
#define CONSOLE_IO_H

#include <iostream>
#include "AMReX_MLMG.H"

namespace amr_wind {
namespace io {

void print_banner(MPI_Comm, std::ostream&);

void print_summary(std::ostream&);

void print_mlmg_header(const std::string&);

void print_mlmg_info(const std::string& solver, const amrex::MLMG& mlmg);
} // namespace io
} // namespace amr_wind

#endif /* CONSOLE_IO_H */
