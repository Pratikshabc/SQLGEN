export interface User {
  id: string;
  username: string;
}

export interface QueryResult {
  question: string;
  query: string;
  result: any[][];
  insights: string;
}

export interface HistoryRecord {
  id: string;
  question: string;
  query: string;
  result: any;
  insights: string;
}