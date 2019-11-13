/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*  SPHERE source code by Anders Damsgaard, 2010-19,                   */
/*  a 3D Discrete Element Method algorithm with CUDA GPU acceleration. */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Licence: GNU Public License (GPL) v. 3. See LICENSE.txt.
// See doc/pdf/sphere.pdf for full documentation.
// Compile with GNU make by typing 'cmake . && make' in the root directory.
// SPHERE is called from the command line with './sphere projectname' 


// Including library files
#include <iostream>
#include <cstdio>
#include <string>
#include <cstdlib>

// Including user files
#include "constants.h"
#include "datatypes.h"
#include "sphere.h"

//////////////////
// MAIN ROUTINE //
//////////////////
// The main loop returns the value 0 to the shell, if the program terminated
// successfully, and 1 if an error occured which caused the program to crash.
int main(const int argc, const char *argv[]) 
{
    // Default values
    int verbose = 1;
    int checkVals = 1;
    int dry = 0;
    int render = 0; // whether to render an image
    int method = 0; // visualization method
    int nfiles = 0; // number of input files
    float max_val = 0.0f;     // max value of colorbar
    float lower_cutoff = 0.0f;// lower cutoff, particles below won't be rendered
    int fluid = 0;
    int device = -1; // -1 run on device with most cores, 0+ run on specified device
    int print_contacts = 0;

    // Process input parameters
    int i;
    for (i=1; i<argc; ++i) {	// skip argv[0]

        std::string argvi = std::string(argv[i]);

        // Display help if requested
        if (argvi == "-h" || argvi == "--help") {
            std::cout << argv[0] << ": particle dynamics simulator\n"
                << "Usage: " << argv[0] << " [OPTION[S]]... [FILE1 ...]\n"
                "Options:\n"
                "-h, --help\t\tprint help\n"
                "-V, --version\t\tprint version information and exit\n"
                "-q, --quiet\t\tsuppress status messages to stdout\n"
                "-d <device>\t\texecute on device with specified id\n"
                "-n, --dry\t\tshow key experiment parameters and quit\n"
                "-f, --fluid\t\tsimulate fluid between particles\n"
                "-r, --render\t\trender input files to images instead of\n"
                "            \t\tsimulating the temporal evolution\n"
                "-dc, --dont-check\tdon't check values before running\n" 
                "\nRaytracer (-r) specific options:\n"
                "-m <method> <maxval> [-l <lower cutoff val>], or\n"
                "--method <method> <maxval> [-l <lower cutoff val>]\n"
                "\tcolor visualization method, possible values:\n"
                "\tnormal, pres, vel, angvel, xdisp, angpos\n"
                "\t'normal' is the default mode\n"
                "\tif -l is appended, don't render particles with value below\n"
                "-c, --contacts\t\tPrint a list of particle-particle contacts\n"
                << std::endl;
            return 0; // Exit with success
        }

        // Display version with fancy ASCII art
        else if (argvi == "-V" || argvi == "--version") {
            printf(
                ".-------------------------------------.\n"
                "|              _                      |\n" 
                "|             | |                     |\n" 
                "|    ___ _ __ | |__   ___ _ __ ___    |\n"
                "|   / __| '_ \\| '_ \\ / _ \\ '__/ _ \\   |\n"
                "|   \\__ \\ |_) | | | |  __/ | |  __/   |\n"
                "|   |___/ .__/|_| |_|\\___|_|  \\___|   |\n"
                "|       | |                           |\n"
                "|       |_|           Version: %.2f   |\n"           
                "`-------------------------------------´\n"
                " A discrete-element method particle dynamics simulator.\n"
                " Written by Anders Damsgaard, license GPLv3+.\n"
                " https://adamsgaard.dk\n", VERSION);
            return 0;
        }

        else if (argvi == "-q" || argvi == "--quiet")
            verbose = 0;

        else if (argvi == "-n" || argvi == "--dry")
            dry = 1;

        else if (argvi == "-r" || argvi == "--render") {
            render = 1;
            checkVals = 0;
        }

        else if (argvi == "-dc" || argvi == "--dont-check")
            checkVals = 0;

        else if (argvi == "-f" || argvi == "--fluid")
            fluid = 1;

        else if (argvi == "-c" || argvi == "--contacts") {
            verbose = 0;
            print_contacts = 1;
        }

        else if (argvi == "-d") {
            device = atoi(argv[i+1]);
            if (device < -1) {
                std::cerr << "Error: The device id must be 0 or larger."
                          << std::endl;
                exit(1);
            }
            i++; // skip ahead
        }

        else if (argvi == "-m" || argvi == "--method") {

            render = 1;

            // Find out which
            if (std::string(argv[i+1]) == "normal")
                method = 0;
            else if (std::string(argv[i+1]) == "pres")
                method = 1;
            else if (std::string(argv[i+1]) == "vel")
                method = 2;
            else if (std::string(argv[i+1]) == "angvel")
                method = 3;
            else if (std::string(argv[i+1]) == "xdisp")
                method = 4;
            else if (std::string(argv[i+1]) == "angpos")
                method = 5;
            else {
                std::cerr << "Visualization method not understood. See `"
                    << argv[0] << " --help` for more information." << std::endl;
                exit(1);
            }

            // Read max. value of colorbar as next argument
            if (method != 0) {
                max_val = atof(argv[i+2]);

                // Check if a lower cutoff value was specified
                if (std::string(argv[i+3]) == "-l") {
                    lower_cutoff = atof(argv[i+4]);
                    i += 4; // skip ahead
                } else {
                    i += 2; // skip ahead
                }
            } else {
                i++;
            }
        }


        // The rest of the values must be input binary files
        else {
            nfiles++;

            if (verbose == 1)
                std::cout << argv[0] << ": processing input file: " << argvi <<
                    std::endl;

            if (print_contacts == 1) {
                DEM dem(argvi, verbose, 0, 0, 0, 0, fluid, device);
                dem.printContacts();
                exit(0);
            }

            if (nfiles == 1) {

                // Create DEM class, read data from input binary, check values,
                // init cuda, transfer const mem
                DEM dem(argvi, verbose, checkVals, dry, 1, 1, fluid, device);

                // Render image if requested
                if (render == 1)
                    dem.render(method, max_val, lower_cutoff);

                // Otherwise, start iterating through time
                else
                    dem.startTime();

            } else { 
                
                // Do not transfer to const. mem after the first file
                DEM dem(argvi, verbose, checkVals, dry, 1, 0, fluid, device);

                if (print_contacts == 1)
                    dem.printContacts();

                // Render image if requested
                else if (render == 1)
                    dem.render(method, max_val, lower_cutoff);

                // Otherwise, start iterating through time
                else
                    dem.startTime();
            }

        }
    }

    // Check whether there are input files specified
    if (!argv[0] || argc == 1 || nfiles == 0) {
        std::cerr << argv[0] << ": missing input binary file\n"
            << "See `" << argv[0] << " --help` for more information"
            << std::endl;
        return 1; // Return unsuccessful exit status
    }

    return 0; // Return successfull exit status
} 
// END OF FILE
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
