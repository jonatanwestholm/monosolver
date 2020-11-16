#include <vector>

using namespace std;

#define Matrix vector<vector<double>>
#define Vector vector<double>

void print_vector(Vector v, bool endl=true){
  int N = v.size();
  int i = 0;
  cout << "[";
  for(auto c : v){
    cout << c;
    if(i < N-1){
      cout << ", ";
    }
    i++;
  }
  cout << "]";
  if(endl){
    cout << "\n";
  }
}

void print_matrix(Matrix m){
  int N = m.size();
  int i = 0;
  cout << "[";  
  for(Vector v : m){
    print_vector(v, false);
    if(i < N-1){
      cout << ",\n";
    }
    i++;
  }
  cout << "]\n";
}

class range {
  public:
    class iterator {
      friend class range;
      public:
        long int operator *() const { return i_; }
        const iterator &operator ++() { ++i_; return *this; }
        iterator operator ++(int) { iterator copy(*this); ++i_; return copy; }

        bool operator ==(const iterator &other) const { return i_ == other.i_; }
        bool operator !=(const iterator &other) const { return i_ != other.i_; }

      protected:
        iterator(long int start) : i_ (start) { }

      private:
        unsigned long i_;
    };

    iterator begin() const { return begin_; }
    iterator end() const { return end_; }
    range(long int  begin, long int end) : begin_(begin), end_(end) {}

  private:
    iterator begin_;
    iterator end_;
};