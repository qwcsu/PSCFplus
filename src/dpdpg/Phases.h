#ifndef PHASES_H
#define PHASES_H

template <class System1, class System2, typename T = double>
class Phases
{

public:

    Phases()=default;
    Phases(System1 sys1, System2 sys2, int argc, char **argv);
    ~Phases() = default;

    T func(T x);

private:
    
    System1 sys1_;
    System2 sys2_;

};

template <class System1, class System2, typename T>
Phases<System1, System2, T>::Phases(System1 sys1, System2 sys2, int argc, char **argv)
{
    
    // sys1.setOptions(argc, argv);

    std::cout << "sys1 have set options\n";
}

template <class System1, class System2, typename T>
T
Phases<System1, System2, T>::func(T x)
{
    return std::exp(x) - 0.5;
}

#endif