import React, { useState, useEffect } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import { Database, Table2, History, Brain, Copy } from 'lucide-react';
import axios from 'axios';
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter';
import sql from 'react-syntax-highlighter/dist/esm/languages/hljs/sql';
import { vs2015 } from 'react-syntax-highlighter/dist/esm/styles/hljs';
import type { User, QueryResult, HistoryRecord } from './types';

SyntaxHighlighter.registerLanguage('sql', sql);

const API_URL = 'http://localhost:8000';

function App() {
  const [user, setUser] = useState<User | null>(null);
  const [tables, setTables] = useState<string[]>([]);
  const [selectedTables, setSelectedTables] = useState<string[]>([]);
  const [question, setQuestion] = useState('');
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [history, setHistory] = useState<Record<string, HistoryRecord>>({});
  const [loading, setLoading] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      // Set axios default header for all requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      // Try to check if token is valid by fetching tables
      fetchTables()
        .then(() => {
          fetchHistory();
          // If token works, we can extract username from localStorage
          const storedUsername = localStorage.getItem('username');
          if (storedUsername) {
            setUser({ id: 'user-id', username: storedUsername });
          }
        })
        .catch(() => {
          // If token doesn't work, clear it
          localStorage.removeItem('token');
          localStorage.removeItem('username');
          delete axios.defaults.headers.common['Authorization'];
        });
    }
  }, []);

  const handleLogin = async () => {
    if (!username.trim()) {
      toast.error('Please enter a username');
      return;
    }

    try {
      setLoading(true);
      const { data } = await axios.post(`${API_URL}/login`, {
        username,
        password: password || username // Use password if provided, otherwise use username as password
      });
      
      // Store token and username
      localStorage.setItem('token', data.token);
      localStorage.setItem('username', username);
      axios.defaults.headers.common['Authorization'] = `Bearer ${data.token}`;
      
      // Set user state
      setUser({ id: 'user-id', username });
      
      toast.success('Successfully logged in!');
      fetchTables();
      fetchHistory();
    } catch (error) {
      console.error('Login error:', error);
      toast.error('Login failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const fetchTables = async () => {
    try {
      const { data } = await axios.get(`${API_URL}/tables`);
      setTables(data.tables);
      return data;
    } catch (error) {
      console.error('Fetch tables error:', error);
      toast.error('Failed to fetch tables');
      throw error;
    }
  };

  const fetchHistory = async () => {
    try {
      const { data } = await axios.get(`${API_URL}/history`);
      setHistory(data.history);
    } catch (error) {
      console.error('Fetch history error:', error);
      toast.error('Failed to fetch history');
    }
  };

  const handleSubmitQuery = async () => {
    if (!question.trim()) {
      toast.error('Please enter a question');
      return;
    }

    setLoading(true);
    try {
      const { data } = await axios.post(
        `${API_URL}/query`,
        { question, selected_tables: selectedTables }
      );
      
      setQueryResult(data);
      await fetchHistory();
      toast.success('Query executed successfully!');
    } catch (error) {
      console.error('Query execution error:', error);
      toast.error('Failed to execute query');
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard!');
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('username');
    delete axios.defaults.headers.common['Authorization'];
    setUser(null);
    setTables([]);
    setSelectedTables([]);
    setHistory({});
    setQueryResult(null);
    toast.success('Signed out successfully');
  };

  const renderHistoryItem = (id: string, record: HistoryRecord) => {
    return (
      <div key={id} className="border rounded-lg p-4">
        <p className="font-medium text-gray-900 mb-2">{record.question}</p>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => copyToClipboard(record.query)}
            className="text-sm text-indigo-600 hover:text-indigo-800 flex items-center space-x-1"
          >
            <Copy className="h-4 w-4" />
            <span>Copy Query</span>
          </button>
          <button
            onClick={() => {
              setQuestion(record.question);
              if (record.result && typeof record.result === 'string') {
                try {
                  // Try to parse the result if it's a string representation
                  const parsedResult = JSON.parse(record.result.replace(/'/g, '"'));
                  setQueryResult({
                    question: record.question,
                    query: record.query,
                    result: parsedResult,
                    insights: record.insights
                  });
                } catch (e) {
                  // If parsing fails, use the original result
                  setQueryResult(record);
                }
              } else {
                setQueryResult(record);
              }
            }}
            className="text-sm text-green-600 hover:text-green-800 flex items-center space-x-1"
          >
            <History className="h-4 w-4" />
            <span>Rerun Query</span>
          </button>
        </div>
      </div>
    );
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="max-w-md w-full space-y-8 bg-white p-8 rounded-xl shadow-lg">
          <div className="text-center">
            <Database className="mx-auto h-12 w-12 text-indigo-600" />
            <h2 className="mt-6 text-3xl font-bold text-gray-900">SQL Query Assistant</h2>
            <p className="mt-2 text-sm text-gray-600">Enter your credentials to continue</p>
          </div>
          <div className="mt-8 space-y-4">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Username"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Password (optional)"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            />
            <button
              onClick={handleLogin}
              disabled={loading}
              className="w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
            >
              {loading ? 'Logging in...' : 'Login'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Toaster position="top-right" />
      
      <nav className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-2">
              <Database className="h-6 w-6 text-indigo-600" />
              <span className="font-semibold text-gray-900">SQL Query Assistant</span>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">Welcome, {user.username}</span>
              <button
                onClick={handleLogout}
                className="text-sm text-red-600 hover:text-red-800"
              >
                Sign out
              </button>
            </div>
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 gap-8">
          {/* Table Selection */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center space-x-2 mb-4">
              <Table2 className="h-5 w-5 text-indigo-600" />
              <h2 className="text-lg font-medium text-gray-900">Available Tables</h2>
            </div>
            <div className="flex flex-wrap gap-2">
              {tables.map((table) => (
                <button
                  key={table}
                  onClick={() => {
                    setSelectedTables((prev) =>
                      prev.includes(table)
                        ? prev.filter((t) => t !== table)
                        : [...prev, table]
                    );
                  }}
                  className={`px-3 py-1 rounded-full text-sm ${
                    selectedTables.includes(table)
                      ? 'bg-indigo-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {table}
                </button>
              ))}
            </div>
          </div>

          {/* Query Input */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="space-y-4">
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Enter your question in natural language..."
                className="w-full h-32 p-3 border rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
              />
              <button
                onClick={handleSubmitQuery}
                disabled={loading}
                className="w-full bg-indigo-600 text-white py-2 px-4 rounded-lg hover:bg-indigo-700 disabled:opacity-50"
              >
                {loading ? 'Processing...' : 'Execute Query'}
              </button>
            </div>
          </div>

          {/* Query Result */}
          {queryResult && (
            <div className="bg-white rounded-lg shadow p-6 space-y-6">
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-medium text-gray-900">Generated SQL Query</h3>
                    <button
                      onClick={() => copyToClipboard(queryResult.query)}
                      className="text-indigo-600 hover:text-indigo-800 flex items-center space-x-1"
                    >
                      <Copy className="h-4 w-4" />
                      <span>Copy</span>
                    </button>
                  </div>
                  <SyntaxHighlighter language="sql" style={vs2015} className="rounded-lg">
                    {queryResult.query}
                  </SyntaxHighlighter>
                </div>

                <div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Result</h3>
                  <div className="overflow-x-auto">
                    {Array.isArray(queryResult.result) && queryResult.result.length > 0 ? (
                      <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                          <tr>
                            {Array.isArray(queryResult.result[0]) && 
                              queryResult.result[0].map((header, i) => (
                                <th
                                  key={i}
                                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                                >
                                  {header}
                                </th>
                              ))}
                          </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                          {queryResult.result.slice(1).map((row, i) => (
                            <tr key={i}>
                              {Array.isArray(row) && 
                                row.map((cell, j) => (
                                  <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                    {cell !== null ? cell.toString() : ''}
                                  </td>
                                ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="text-gray-700">No results to display</p>
                    )}
                  </div>
                </div>

                <div>
                  <div className="flex items-center space-x-2 mb-2">
                    <Brain className="h-5 w-5 text-indigo-600" />
                    <h3 className="text-lg font-medium text-gray-900">Insights</h3>
                  </div>
                  <p className="text-gray-700 whitespace-pre-wrap">{queryResult.insights}</p>
                </div>
              </div>
            </div>
          )}

          {/* History */}
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center space-x-2 mb-4">
              <History className="h-5 w-5 text-indigo-600" />
              <h2 className="text-lg font-medium text-gray-900">Query History</h2>
            </div>
            <div className="space-y-4">
              {Object.keys(history).length > 0 ? (
                Object.entries(history).map(([id, record]) => renderHistoryItem(id, record))
              ) : (
                <p className="text-gray-500 text-center py-4">No query history yet</p>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;