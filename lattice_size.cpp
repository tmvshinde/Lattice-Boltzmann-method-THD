

#include <iostream>

using namespace std;

class Lattice_inputs{

    public:
        void set_lattice_size(int lattice_size);
        int get_lattice_size( void );
        Lattice_inputs();
    
    private:
        int lattices;
};

Lattice_inputs::Lattice_inputs(void){
    // cout << "Enter the number of lattice in x - direction" << endl;
}

void Lattice_inputs::set_lattice_size(int lattice_size){
    lattices = lattice_size;
}

int Lattice_inputs::get_lattice_size(void){
    return lattices;
}

int main(){
    Lattice_inputs lattice;

    lattice.set_lattice_size(16);

    int lattice_size = lattice.get_lattice_size();

    cout << "Number of lattices : " << lattice_size << endl;

    return 0;
}