// App.jsx

import React from 'react';
import { useState, useEffect } from 'react';
import ChatComponent from './ChatComponent';

const App = () => {
  return (
    <div className="app">
      <ChatComponent />
    </div>
  );
};

export default App;