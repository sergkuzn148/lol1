#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>

int print_usage(FILE* stream, char* argv0, int return_status);
int open_status_file(char* cwd, char* sim_name, int format);

int main(int argc, char *argv[])
{

    // Read path to current working directory
    char *cwd;
    cwd = getcwd(0, 0);
    if (!cwd) {  // Terminate program execution if path is not obtained
        perror("Could not read path to current workind directory "
                "(getcwd failed)");
        return 1;
    }

    if (argc == 1 || argc != 2) {
        return print_usage(stderr, argv[0], 1);
    } else if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
        return print_usage(stdout, argv[0], 0);
    } else if (strcmp(argv[1], "-l") == 0 || strcmp(argv[1], "--list") == 0) {
        struct dirent **namelist;
        int n, i;
        char outputdir[1000];
        char* dotpos;
        char outstring[100];
        char* p;
        snprintf(outputdir, sizeof(outputdir), "%s/output/", cwd);
        n = scandir(outputdir, &namelist, 0 , alphasort);
        if (n < 0) {
            fprintf(stderr, "Error: could not open directory: %s\n", outputdir);
            return 1;
        } else {
            puts("Simulations with the following ID's are found in the "
                    "./output/ folder:");

            for (i = 0; i<n; i++) {
                if ((dotpos = strstr(namelist[i]->d_name, ".status.dat"))
                        != NULL) {
                
                    *dotpos = '\0';
                    snprintf(outstring, sizeof(outstring), "%-54s ",
                             namelist[i]->d_name);
                    for (p = outstring; *p != '\0'; p++)
                        if (*p == ' ') *p = '.';
                    printf("  %s", outstring);
                    (void)open_status_file(cwd, namelist[i]->d_name, 1);
                    puts("");
                }
                free(namelist[i]);
            }
            free(namelist);
        }
        return 0;

    }
    int ret = open_status_file(cwd, argv[1], 0);
    free(cwd);
    return ret;
}

int print_usage(FILE* stream, char* argv0, int return_status)
{
    fprintf(stream, "sphere simulation status checker. Usage:\n"
            " %s [simulation id]\n"
            " %s [-h,--help]\n"
            " %s [-l,--list]\n"
            "Arguments:\n"
            " simulation id\tShow detailed status of simulation.\n"
            " -h, --help\tShow this help message.\n"
            " -l, --list\tPrint a list of simulations found in the ./output/ "
            "folder.\n"
            "\t\tEach simulation ID will be appended by a string showing:\n"
            "\t([CURRENT SIMULATION TIME] s, [PERCENTAGE COMPLETED] %%, "
            "[LATEST OUTPUT FILE])\n", argv0, argv0, argv0);
    return return_status;
}


int open_status_file(char* cwd, char* sim_name, int format) {
    // Open the simulation status file
    FILE *fp;
    char file[1000]; // Complete file path+name variable
    snprintf(file, sizeof(file), "%s/output/%s.status.dat", cwd, sim_name);

    if ((fp = fopen(file, "rt"))) {
        float time_current;
        float time_percentage;
        unsigned int file_nr;

        if (fscanf(fp, "%f%f%d", &time_current, &time_percentage, &file_nr)
                != 3) {
            fprintf(stderr, "Error: could not parse file %s\n", file);
            return 1;
        }

        if (format == 1) {
            printf("%6.2fs / %3.0f%% / %5d",
                    time_current, time_percentage, file_nr);
        } else {
            printf("Reading %s:\n"
                    " - Current simulation time:  %f s\n"
                    " - Percentage completed:     %f %%\n"
                    " - Latest output file:       %s.output%05d.bin\n",
                    file, time_current, time_percentage, sim_name, file_nr);
        }

        fclose(fp);

        return 0; // Exit program successfully

    } else {
        fprintf(stderr, "Error: Could not open file %s\n"
                "Run this program with `--help` flag for help.\n", file);
        return 1;
    }
}
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
