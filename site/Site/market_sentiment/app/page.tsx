'use client';

import { Dispatch, SetStateAction, useEffect, useState } from 'react';
import './globals.css'

const defaultCompanies = ["meta", "amazon", "apple", "netflix", "google"];
const requestUrl = "http://127.0.0.1:5000/sentiment_score";

function onEnter(e: React.KeyboardEvent<HTMLInputElement>, 
                    defaultElems: Array<React.JSX.Element>, 
                    setDefaultElems: Dispatch<SetStateAction<JSX.Element[]>>) {
  if (e.key === "Enter") {
    const ticker = (e.target as HTMLInputElement).value;
    addCompanies([ticker], defaultElems, setDefaultElems);
  }
}

function addCompanies(tickers: Array<string>, defaultElems: Array<React.JSX.Element>, setDefaultElems: Dispatch<SetStateAction<JSX.Element[]>>) {
  tickers.forEach(ticker => {
    fetch(requestUrl + "/" + ticker, {method: "POST"})
    .then(response => response.json())
    .then(response => {
      let sentimentScore = response["sentiment score"];
      let urls = response["articles"];
      let newElem = <div className="flex-none text-center rounded-md border-white border-4 p-5 m-5" key={defaultElems.length}>
        <label>{ticker}</label>
        <br></br>
        <label>{sentimentScore}</label>
      </div>;

      defaultElems = [...defaultElems];
      defaultElems.push(newElem);
      setDefaultElems(defaultElems);
    });
  });
}

export default function Home() {
  const [defaultElems, setDefaultElems] = useState<Array<React.JSX.Element>>([]);
  const [textField, setText] = useState<string>("");


  useEffect(() => {
    addCompanies(defaultCompanies, defaultElems, setDefaultElems);
  }, []);

  if (defaultElems.length === 0) {
    return (
      <main className="flex min-h-screen flex-col items-center justify-between p-24">
        Loading
      </main>
    )
  } else {
    return (
      <main className="flex min-h-screen flex-col items-center justify-between p-24">
        <div className="flex gap-x-20 flex-wrap justify-center">
          {defaultElems}
        </div>
        <div className="flex flex-col items-center justify-center">
          <input type="text" className="block w-full p-4 ps-10 text-sm text-gray-900 border border-gray-300 rounded-lg bg-gray-50 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500" onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => {onEnter(e, defaultElems, setDefaultElems); setText(e.currentTarget.value + e.key);}}></input>
          <button type="submit" className="m-2 items-center text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:outline-none focus:ring-blue-300 font-medium rounded-lg text-sm px-4 py-2 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800" onClick={(e) => {addCompanies([textField], defaultElems, setDefaultElems)}}>Add Subject</button>
        </div>
      </main>
    )
  }
}
