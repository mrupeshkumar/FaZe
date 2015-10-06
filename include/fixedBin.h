#ifndef FIXED_BIN_H
#define FIXED_BIN_H
#endif

template <class T>
class FixedBin {

private:
	std::vector< T > _bin;
	int _size;
	int _filled;

public:
	FixedBin();
	void assign(int size);
	void push(T t);
	int size();
	int filled();
	T get(int pos);
	std::vector< T > clone();
};

template <class T> FixedBin< T >::FixedBin() {
	_bin = std::vector< T >(0);
}

template <class T> void FixedBin< T >::assign(int size) {
	_filled = 0;
	_size = size;
	_bin.resize(_size);
}

template <class T> void FixedBin< T >::push(T t) {
	if(_filled == _size) {
		for(int i=0;i<_filled-1;i++) {
			_bin[i] = _bin[i+1];
		}
		_bin[_filled - 1] = t;
	}
	else {
		_bin.at(_filled) = t;
		++_filled;
	}
}

template <class T> int FixedBin< T >::size() {
	return _size;
}

template <class T> int FixedBin< T >::filled() {
	return _filled;
}

template <class T> T FixedBin< T >::get(int pos) {
	return _bin.at(pos);
}

template <class T> std::vector< T > FixedBin< T >::clone() {
	std::vector< T > vec(_filled);
	for(int i=0;i<_filled;i++) {
		vec.at(i) = _bin.at(i);
	}
	return vec;
}